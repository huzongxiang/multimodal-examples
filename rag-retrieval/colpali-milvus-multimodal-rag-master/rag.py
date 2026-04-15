import requests
import os
from pathlib import Path

from typing import List
from dotenv import load_dotenv
from utils import encode_image

load_dotenv(Path(__file__).with_name(".env"))

class Rag:

    def get_answer_from_glm(self, query, image_paths):
        print(f"Querying GLM for query={query}, image_paths={image_paths}")

        try:
            payload = self.__get_glm_api_payload(query, image_paths)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['ZHIPU_API_KEY']}",
            }

            response = requests.post(
                url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            if not response.ok:
                return f"Error: GLM API returned {response.status_code}: {response.text}"

            answer = response.json()["choices"][0]["message"]["content"]

            print(answer)

            return answer

        except Exception as e:
            print(f"An error occurred while querying GLM: {e}")
            return f"Error: {str(e)}"
        

    def get_answer_from_openai(self, query, imagesPaths):
        print(f"Querying OpenAI for query={query}, imagesPaths={imagesPaths}")

        try:    
            payload = self.__get_openai_api_payload(query, imagesPaths)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
            }
    
            response = requests.post(
                url="https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()  # Raise an HTTPError for bad responses
    
            answer = response.json()["choices"][0]["message"]["content"]
    
            print(answer)
    
            return answer
    
        except Exception as e:
            print(f"An error occurred while querying OpenAI: {e}")
            return None


    def __get_openai_api_payload(self, query:str, imagesPaths:List[str]):
        image_payload = []

        for imagePath in imagesPaths:
            base64_image = encode_image(imagePath)
            image_payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        },
                        *image_payload
                    ]
                }
            ],
            "max_tokens": 1024
        }

        return payload


    def __get_glm_api_payload(self, query: str, image_paths: List[str]):
        image_payload = []

        for image_path in image_paths:
            base64_image = encode_image(image_path)
            image_payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        payload = {
            "model": os.getenv("GLM_MODEL", "glm-4.6v-flash"),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *image_payload,
                        {
                            "type": "text",
                            "text": query
                        }
                    ]
                }
            ],
            "max_tokens": 1024
        }

        return payload
    


# if __name__ == "__main__":
#     rag = Rag()
    
#     query = "Based on attached images, how many new cases were reported during second wave peak"
#     imagesPaths = ["covid_slides_page_8.png", "covid_slides_page_8.png"]
    
#     rag.get_answer_from_glm(query, imagesPaths)
