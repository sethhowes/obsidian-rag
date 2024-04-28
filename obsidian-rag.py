import os
from argparse import ArgumentParser

from langchain import hub
from langchain_community.document_loaders import ObsidianLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

parser = ArgumentParser(description="Query your Obsidian knowledgebase.")
subparsers = parser.add_subparsers(help="Select the command to run", dest="command")
query_parser = subparsers.add_parser("query", help="Enter the query as a string")
query_parser.add_argument("query_string", nargs="?", help="Enter the query as a string")
sync_parser = subparsers.add_parser("sync", help="Sync your database")
args = parser.parse_args()

obsidian_folder = os.path.join(os.path.expanduser("~"), ".obsidian-rag")
embedding_client = (
    OpenAIEmbeddings(organization=os.environ.get("OPENAI_ORGANIZATION_ID"))
    if os.environ.get("OPENAI_ORGANIZATION_ID")
    else OpenAIEmbeddings()
)

if args.command == "sync":
    print("Syncing your Obsidian database...")
    loader = ObsidianLoader("/Users/sethhowes/Desktop/obsidian")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)[:100]

    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embedding_client,
        persist_directory=obsidian_folder,
    )

    print("Database synced!")


if args.command == "query":
    vectorstore = Chroma(
        persist_directory=obsidian_folder, embedding_function=embedding_client
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

    for chunk in rag_chain.stream(args.query_string):
        print(chunk, end="", flush=True)
