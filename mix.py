import numpy as np
import pandas as pd
import streamlit as st
from streamlit import caching 
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from summarizer import Summarizer
from sumy.parsers.plaintext import PlaintextParser

#App uses two NLP models to summarzie text. The larger and slower BART model is downloaded once the app runs and is cached. Caching will enable you to 
#test the trained BART model on as much tesxt as is needed without having to upload the model for each test. The Lex_Rank model is much smaller and faster.

# Loading the model and tokenizer for bart-large-cnn
@st.cache(allow_output_mutation=True)
def get_model():
	tokenizer=BartTokenizer.from_pretrained('facebook/bart-large-cnn')
	model1=BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
	return tokenizer,model1
tokenizer,model1= get_model() 

#Use pretrained model and tokenizer to produce summary tokens and then decode these summarized tokens 
@st.cache(allow_output_mutation=True)
def summarizer(original_text):
	inputs = tokenizer.batch_encode_plus([original_text],return_tensors='pt')
	summary_ids = model1.generate(inputs['input_ids'], early_stopping=True)
	bart_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
	return bart_summary

# Loading the model and tokenizer for bert model
@st.cache(allow_output_mutation=True)
def get_model():
    model = Summarizer()
    return model()
model=get_model
	

#Use pretrained BERT MODEL 
@st.cache(allow_output_mutation=True)
def bert_summarizer(original_text):
    model = Summarizer()
    bert_summary = model(original_text, min_length=20,max_length=500)
    return bert_summary


def main():
	""" NLP Based App with Streamlit. Choice of two mdodels. BART Large or BERT Model """

	# Title
	st.title("Text Summarizer")
	st.write("#")

	 
	# Summarization	
	message = st.text_area("Enter Text in box below")
	st.write('#')

	#provide choice of model. 
	col1, col2 = st.columns(2)
	with col1:
		bart_button=st.button('Summarzier with BART Model')
		if bart_button:
			summary_result = summarizer(message)
			st.success(summary_result)
	with col2:
		bert_button=st.button('Summarizer with BERT Model')
		if bert_button:
			summary_result = bert_summarizer(message)
			st.success(summary_result)

if __name__ == '__main__':
	main()
