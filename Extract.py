
import torch
import requests
from lavis.models import load_model_and_preprocess, load_model


# In[3]:


from PIL import Image


# In[4]:


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


# In[5]:


BLIPCAP, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
)


# In[18]:


image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)


# In[22]:


import openai
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import cv2
import matplotlib.pyplot as plt 
from num2words import num2words
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

#your open ai key
openai.api_key = ""
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
#aspa = adapative summary-prompt answering

@retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(7), reraise=True)
def aspa (image, question):
    from PIL import Image
    raw_image = Image.open(image).convert('RGB')
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    with torch.no_grad():
        max_length = 60
        length_penalty = 1
        repetition_penalty = 1.5
        temperature=1


        #Basic captions
        caption = BLIPCAP.generate({"image": image, 
            "prompt": '''Explain the photo descriptively. Let's work this out in a step by step way to be sure we have the right answer. Answer:'''}, 
            use_nucleus_sampling=True, 
            max_length=max_length, 
            length_penalty=length_penalty, 
            repetition_penalty=repetition_penalty, 
            temperature=temperature,
            num_captions=3)

        #Descriptive caption 
        sum = ""
        ppl = BLIPCAP.generate({
            "image": image, 
            "prompt": "Question: How many people are there? Answer:"
            },
            use_nucleus_sampling=True, 
            max_length=max_length, 
            length_penalty=length_penalty, 
            repetition_penalty=repetition_penalty, 
            temperature=temperature, 
            num_captions = 1
            )[0]
        #Location
        where = BLIPCAP.generate({
            "image": image, 
            "prompt": '''Question: Where was this photo taken specifically? Let's work this out in a step by step way to make sure we have the right answer. Answer:'''
            },
            use_nucleus_sampling=True, 
            max_length=max_length, 
            length_penalty=length_penalty, 
            repetition_penalty=repetition_penalty, 
            temperature=temperature, 
            num_captions = 1    
            )[0]
        #What is happening?
        what = BLIPCAP.generate({
            "image": image, 
            "prompt": "Question: What is happening? Let's work this out in a step by step way to be sure we have the right answer. Answer:"
            },
            use_nucleus_sampling=True, 
            max_length=max_length, 
            length_penalty=length_penalty, 
            repetition_penalty=repetition_penalty, 
            temperature=temperature, 
            num_captions = 1
            )[0]

    #     if (ppl[0]!=('0')):
    #         emotions = BLIPCAP.generate({
    #             "image": image, 
    #             "prompt": "Question: What emotions do they feel? Work this out in a step by step way to be sure you have the right answer. Answer:"}, 
    #             use_nucleus_sampling=True, 
    #             max_length=max_length, 
    #             length_penalty=length_penalty, 
    #             repetition_penalty=repetition_penalty, 
    #             temperature=temperature, 
    #             num_captions = 1)[0]

        objectlist = BLIPCAP.generate({
            "image": image, 
            "prompt": "Question: List all objects in the photo. Let's think step by step. Answer:"
            },
            use_nucleus_sampling=True, 
            max_length=max_length, 
            length_penalty=length_penalty, 
            repetition_penalty=repetition_penalty, 
            temperature=temperature, 
            num_captions = 1
            )[0]
        
        a = BLIPCAP.generate({
            "image": image, 
            "prompt": question,
            },
            use_nucleus_sampling=True, 
            max_length=max_length, 
            length_penalty=length_penalty, 
            repetition_penalty=repetition_penalty, 
            temperature=temperature, 
            num_captions = 1
            )[0]
        
        
        com=-1
        objs=""
        for i in reversed(range(len(objectlist))):
            if objectlist[i]==",":
                com = i
                break
        for i in (range(len(objectlist))):
            objs+=objectlist[i]
            if i == com:
                objs+=" and"

        cap_0=f"{caption[0]}"
        cap_1=f"{caption[1][0].upper()}{caption[1][1:]}"
        cap_2=f"{caption[2][0].upper()}{caption[2][1:]}"
        
        

        sum = f"{cap_0} {cap_1} {cap_2} There are {ppl} people. {what}. {objs}. {question} {a}" 

        print(f"\nIn summary: {sum}\n")

        question_nochoices = question.split("\na.")[0]

        generate_questions = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=[
                    {"role": "system", "content": "You are an intelligent and helpful assistant."},
                    {"role": "user", "content": "Describe the image in great detail."},
                    {"role": "assistant", "content": f"{sum}"},
                    {"role": "user", "content": f"Generate 10 follow-up questions that are important to answer {question} Let's think step by step. Answer:"}

                ]
            )

        questions = generate_questions["choices"][0]["message"]["content"]
        print (questions)
        #.strip().split("\n")
        answers = []
        for q in questions:
            q = q.strip("-").replace("-", "")
            answer = BLIPCAP.generate({"image": image, "prompt": f"Question: {q} Answer:"})[0]
            answers.append(answer)
        qa_pairs = [f"Q: {question.strip('-')}\nA: {answer}" for question, answer in zip(questions,answers)]
        qa_pairs_str = "\n".join(qa_pairs)
        #example = "I see a photo of a person on a beach playing volleyball. I ask the following questions about it and received the following answers:\n\nQ: What color is the ball?\nA: Blue and yellow\nQ: How many people are there?\nA: Five people.\n\nWhy is one of the people raising their hands?\na. They are giving someone a high-five.\nb. They are spiking the ball.\nc. They are drinking wine.\nd. They are cheering on their teammates.\nAnswer: High-fiving only requires one hand, so the answer is not a. Similarly, spiking the ball requires only one hand, so the answer is not b. Drinking wine is not typically done in the middle of a volleyball game, so the answer is not c. Cheering on teammates could require both hands, so the most plausible answer is d. Final answer: d"
        #example2 = "I see a photo of a glass of water with ice cubes in it. I ask the following questions about it and received the following answers:\n\nQ: How full is the glass?\nA: The glass is half full.\nQ: Does the glass of water have a straw?\nA: No, the glass of water does not have a straw.\n\nWhy is the outside of the glass wet?\na. The light causes it to get wet \nb. The ice water cools the glass, causing moisture in the air to condense onto the outer surface. \nc. The glass is in the middle of the ccean.\nd. The water is spilling over the glass.\nAnswer: Shining light on an object does not make it wet, so the answer is not a. A glass in the middle of the ocean is unlikely to be photographed, so the answer is not c. The glass is not stated to be spilling over, so the answer is not d. Water vapor does indeed condense onto cold surfaces, so the best answer is b. Final answer: b"

        prompt_summarize = f'''\nKnowledge: {qa_pairs_str}\nQuestion: Using your knowledge, summarize the information accurately and concisely. Work this out in a step by step way to be sure you have the right answer.\nAnswer: '''

        question_sum = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=[
                    {"role": "system", "content": "You are an intelligent and helpful assistant."},
                    {"role": "user", "content": prompt_summarize},
                ]
            )
    {question_sum["choices"][0]["message"]["content"]}.  

        result = f'''There is a photo. In this photo, {sum} Use common sense and reasoning with the given information to answer "{question}" 
Report the most probable answer to this question and your confidence on this answer, where your concise answer is in 'Answer' and your confidence regarding your answer is in 'Confidence'. 
Your 'Answer' must be one word only. Your 'Confidence' must only be "yes" or "maybe". If you do not know the answer, you should guess the answer. 
'''
        output = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          temperature=0.8,
          messages=[
                {"role": "system", "content": "You must use your reasoning capabilities to answer questions correctly."}, 
                {"role": "user", "content": result},
            ]
        )
        ans = output["choices"][0]["message"]["content"].strip()
        
            
        print(ans)
        ans_list = ans.lower().replace(":", "").replace(".", "").split()
        Answer = ""
        Confidence = ""
        for i in range(len(ans_list)):
            if ans_list[i] == 'answer':
                b = i
                while b+1 < len(ans_list) and ans_list[b+1] != 'confidence':
                    if (ans_list[b+1].isnumeric()): ans_list[b+1] = num2words(ans_list[b+1])
                    Answer += ans_list[b+1] + " "
                    b+=1
            if i+1 < len(ans_list) and ans_list[i] == 'confidence':
                Confidence = ans_list[i+1]
        Answer = Answer.strip()
        if(Answer==""): 
            Answer = ans[:ans.find(".")].lower()
        if (Confidence == 'high'): Confidence = 'yes'
        elif (Confidence.find("%") != -1):
            if (int(Confidence[:2]) >= 50): Confidence = 'yes'
            else: Confidence = 'maybe'
        elif (Confidence.find("50/50") != -1):
            Confidence = 'yes'
        elif (Confidence.find("low") != -1):
            Confidence = 'maybe'
        ans = f"{Answer}_{Confidence}"
        return ans


# In[26]:


import os
import json
import random 
random.seed(42)
counter = 0
num_images = 40504
answers_list=[]
answer=""
confidence=""
size = 40504
with open("/workspace/VQAv2/Questions/v2_OpenEnded_mscoco_val2014_questions.json") as questions:
    with open("/workspace/VQAv2/Annotations/v2_mscoco_val2014_annotations.json") as annotations:
        annots = annotations.readlines()
        annos = json.loads(annots[0])
        quest = questions.readlines() 
        qs = json.loads(quest[0])
        #for i in tqdm(range(20+1, size)):
        #wherever it breaks +1
        for i in tqdm(range(size)):
            img_fn = qs["questions"][i]["image_id"]
            if (len(str(img_fn))<6): continue
            image = f"/workspace/VQAv2/Images/val2014/COCO_val2014_000000{img_fn}.jpg"
            question = qs["questions"][i]["question"]
            ans = aspa(image, question)
            print(ans)
            for x in range(10):
                answer = annos["annotations"][i]["answers"][x]["answer"]
                confidence = annos["annotations"][i]["answers"][x]["answer_confidence"]
                reply = ans[:ans.index("_")]
                conf = ans[ans.index("_")+1:len(ans)]
                flag = False 
                if reply == answer:
                    flag = False
                    if conf == "yes":
                        counter+=1
                        flag = True 
                        break
                    elif conf == "maybe":
                        counter+=0.5
                        flag = True 
                        break
                    if (flag): break
                if (flag): break
        if (i+1)%1000 == 0: print(i, counter/size)
print(counter/size)

