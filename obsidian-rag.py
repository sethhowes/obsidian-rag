import os
from argparse import ArgumentParser

from langchain import hub
from langchain_community.document_loaders import ObsidianLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.indexes import SQLRecordManager, index

from dotenv import load_dotenv

load_dotenv()

parser = ArgumentParser(description="Query your Obsidian knowledgebase.")
subparsers = parser.add_subparsers(help="Select the command to run", dest="command")
query_parser = subparsers.add_parser("query", help="Enter the query as a string")
query_parser.add_argument("query_string", nargs="?", help="Enter the query as a string")
sync_parser = subparsers.add_parser("sync", help="Sync your database")
sync_parser.add_argument(
    "--verbose", "-v", action="store_true", help="Print debug information"
)
args = parser.parse_args()

collection_name = "obsidian-rag"
obsidian_db_path = os.path.join(os.path.expanduser("~"), ".obsidian-rag")
obsidian_dir = "/Users/sethhowes/Desktop/obsidian"
embedding_client = (
    OpenAIEmbeddings(organization=os.environ.get("OPENAI_ORGANIZATION"))
    if os.environ.get("OPENAI_ORGANIZATION")
    else OpenAIEmbeddings()
)


def split_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    return text_splitter.split_documents(docs)


if args.command == "sync":
    loader = ObsidianLoader(obsidian_dir, collect_metadata=False)
    docs = loader.load()
    all_splits = split_docs(docs)[:500]
    vectorstore = Chroma(
        embedding_function=embedding_client,
        persist_directory=obsidian_db_path,
    )
    print("Syncing your Obsidian database...")

    embedding = OpenAIEmbeddings()

    # Setup the record manager for recording which documents have been indexed
    namespace = f"chroma/{collection_name}
    record_manager_path = os.path.join(
        os.path.expanduser("~"), ".obsidian-rag", "record_manager_cache.sql"
    )
    record_manager = SQLRecordManager(
        namespace, db_url=f"sqlite:///{record_manager_path}"
    )

    # Create the schema if it doesn't exist
    if not os.path.exists(record_manager_path):
        record_manager.create_schema()

    # Index the documents
    result = index(
        all_splits,
        record_manager,
        vectorstore,
        cleanup="incremental",
        source_id_key="source",
    )
    print(result)
    print("Database synced!")


if args.command == "query":
    vectorstore = Chroma(
        persist_directory=obsidian_db_path, embedding_function=embedding_client
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )
    retrieved_docs = retriever.invoke(args.query_string)

    llm = ChatOpenAI(model="gpt-4-turbo")
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "/n/n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Passes the query string to the chain and prints response stream as it comes in
    for chunk in rag_chain.stream(args.query_string):
        print(chunk, end="", flush=True)
