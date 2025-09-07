import os

from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError

load_dotenv()


class OpenAiClient:
    def __init__(self):
        self.api_key = ""

    def create(self) -> OpenAI | None:
        try:
            client = OpenAI(api_key=self.api_key)
            print(client.models.list())
            return client
        except AuthenticationError:
            print("Invalid OpenAI API key.")
        return None


if __name__ == '__main__':
    openai_client = OpenAiClient()
    client = openai_client.create()
    client.completions.create(model="gpt-5", prompt="Please let me knw the capital  of india ")
    if client:
        print("OpenAI client created successfully.")
    else:
        print("Failed to create OpenAI client.")
