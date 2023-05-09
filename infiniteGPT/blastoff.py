import argparse
import openai
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor

# Add your own OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

def load_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def save_to_file(responses, output_file):
    with open(output_file, 'w') as file:
        for response in responses:
            file.write(response + '\n')

def call_openai_api(task, chunk):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Please {task} the following: {chunk}."},
            ],
            max_tokens=1750,
            n=1,
            stop=None,
            temperature=0.5,
        )
        return response.choices[0]['message']['content'].strip()
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return ""

def split_into_chunks(text, tokens=1500):
    words = text.split()
    chunks = [' '.join(words[i:i + tokens]) for i in range(0, len(words), tokens)]
    return chunks

def process_chunks(task, input_file, output_file):
    text = load_text(input_file)
    chunks = split_into_chunks(text)
    
    # Processes chunks in parallel
    with ThreadPoolExecutor() as executor:
        responses = list(tqdm(executor.map(lambda chunk: call_openai_api(task, chunk), chunks), total=len(chunks), desc="Processing chunks"))
    save_to_file(responses, output_file)

def main():
    parser = argparse.ArgumentParser(description='Process text chunks with OpenAI API.')
    parser.add_argument('task', type=str, help='Task to perform, e.g., "summarize" or "translate".')
    parser.add_argument('input_file', type=str, help='Path to the input text file.')
    parser.add_argument('output_file', type=str, help='Path to the output text file.')

    args = parser.parse_args()

    process_chunks(args.task, args.input_file, args.output_file)

if __name__ == "__main__":
    main()
