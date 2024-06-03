from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
from langchain.chat_models import ChatOpenAI
import gradio
import os

os.environ["OPENAI_API_KEY"] = ""

def construct_index(directory_path):
    # set number of output tokens
    num_outputs = 256

    _llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    service_context = ServiceContext.from_defaults(llm_predictor=_llm_predictor)

    docs = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)

    #Directory in which the indexes will be stored
    index.storage_context.persist(persist_dir="indexes")

    return index

def chatbot_send_request(input_text):

    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="indexes")

    #load indexes from directory using storage_context
    query_engne = load_index_from_storage(storage_context).as_query_engine()

    response = query_engne.query(input_text)

    #returning the response
    return response.response

#Creating the web UIusing gradio
# iface = gradio.Interface(fn=chatbot_send_request,
#                          inputs=gradio.components.Textbox(lines=5, label="Enter your question about Edinburgh new STL policies here."),
#                          outputs="text",
#                          title="AI Chatbot trained to advice on the new STL regulations introduced by Edinburgh")


with (gradio.Blocks() as demo):
    chatbot = gradio.Chatbot(height=750, layout='panel')
    msg = gradio.Textbox( label='Ask your questions here...', value='Hi, What do you know about the new Edinburgh STL policies?')
    clear = gradio.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        bot_message = chatbot_send_request(message)
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

#Constructing indexes based on the documents in traininData folder
#This can be skipped if you have already trained your app and need to re-run it
index = construct_index("/Users/eyeyeye/Downloads/stl")

#launching the web UI using gradio
# iface.launch(share=True)
demo.launch()