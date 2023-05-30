from pathlib import Path
from langchain import OpenAI
from llama_index import GPTVectorStoreIndex, ServiceContext, StorageContext
from llama_index.llm_predictor import LLMPredictor
from llama_index import SimpleDirectoryReader, GPTSimpleKeywordTableIndex, download_loader
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.query_engine.transform_query_engine import TransformQueryEngine
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# wiki_titles = ["Mumbai", "Delhi", "Bhopal", "Ahmedabad", "Chennai"]

wiki_titles = ["Bangladesh", "China", "India"]

PandasExcelReader = download_loader("PandasCSVReader")

loader = PandasExcelReader()
doc = loader.load_data(file=Path(f'data/US.csv'))

city_docs = {}

city_docs["United States"] = doc

for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(input_files=[f"data/{wiki_title}.pdf"]).load_data()

wiki_titles.append('United States')
# set service context
llm_predictor_gpt4 = LLMPredictor(llm=OpenAI(temperature=1, model_name="text-davinci-003"))
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor_gpt4, chunk_size_limit=1024
)

# Build document index
vector_indices = {}
for wiki_title in wiki_titles:
    storage_context = StorageContext.from_defaults()
    # build vector index
    vector_indices[wiki_title] = GPTVectorStoreIndex.from_documents(
        city_docs[wiki_title],
        service_context=service_context,
        storage_context=storage_context,
    )
    # set id for vector index
    vector_indices[wiki_title].index_struct.index_id = wiki_title
    # persist to disk
    storage_context.persist(persist_dir=f'./storage/{wiki_title}')

index_summaries = {}
for wiki_title in wiki_titles:
    # set summary text for city
    index_summaries[wiki_title] = (
        f"This content contains articles about {wiki_title}. "
        f"Use this index if you need to lookup specific facts about {wiki_title}.\n"
        "Do not use this index if you want to analyze multiple countries."
    )

graph = ComposableGraph.from_indices(
    GPTSimpleKeywordTableIndex,
    [index for _, index in vector_indices.items()],
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk=50
)

# get root index
root_index = graph.get_index(graph.index_struct.index_id)
# set id of root index
root_index.set_index_id("compare_contrast")
root_summary = (
    "This index contains articles about multiple countries. "
    "Use this index if you want to compare multiple countries. "
)

# define decompose_transform
decompose_transform = DecomposeQueryTransform(
    llm_predictor_gpt4, verbose=True
)

# define custom query engines
custom_query_engines = {}
for index in vector_indices.values():
    query_engine = index.as_query_engine(service_context=service_context)
    query_engine = TransformQueryEngine(
        query_engine,
        query_transform=decompose_transform,
        transform_extra_info={'index_summary': index.index_struct.summary}
    )
    custom_query_engines[index.index_id] = query_engine


custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
    retriever_mode='simple',
    response_mode='tree_summarize',
    service_context=service_context,
)

# define query engine
query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)

# query the graph
query_str = (
    "which countries have a tax treaty with the United States?"
    #
    # "do australia have a bilateral agreement with the US?"
)

response_chatgpt = query_engine.query(query_str)
print(response_chatgpt)
print("token usage")
print(llm_predictor_gpt4.last_token_usage)