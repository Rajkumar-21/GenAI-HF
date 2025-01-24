import os
import pandas as pd
import yfinance as yf
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

# Load the tokenizer and model for question answering
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

# Function to answer questions given a context
def answer_question(question, context):
    # Tokenize input question and context
    inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Model output
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    # Find the position (tokens) of the start and end of the answer
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    # Convert tokens to the answer string
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))
    return answer

# Function to get stock data and answer questions about it
def get_stock_info_and_answer(stock_ticker, question):
    # Download stock data
    stock_data = yf.download(stock_ticker, period="5d")
    closing_prices = stock_data['Close']

    # Create context from the stock data
    context = f"The closing prices for {stock_ticker} over the last 5 days were as follows: "
    # Use 'Date' directly from the row after resetting the index
    context += ' '.join([f"On {row['Date'].strftime('%Y-%m-%d')} the closing price was {row['Close']:.2f}." for _, row in closing_prices.reset_index().iterrows()])

    # Answer the question using the context
    answer = answer_question(question, context)
    return answer

# Example usage
stock_ticker = "MSFT"
question = "What was the closing price of MSFT on the last day?"
answer = get_stock_info_and_answer(stock_ticker, question)
print(f"Question: {question}")
print(f"Answer: {answer}")

