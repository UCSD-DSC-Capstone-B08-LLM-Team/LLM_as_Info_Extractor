# LLM_as_Info_Extractor

## Medical Needle Dataset Generator
A Python tool for generating medical notes with hidden conditions to test Large Language Model capabilities in medical information extraction and reasoning.

## Overview
This tool creates realistic medical notes that contain subtle clues about specific medical conditions without explicitly stating them. It then evaluates whether LLMs can correctly identify these hidden conditions. Our goal is to test the LLM on real data and to see whether LLM's can be used in the real world.

## Usage Instruction
To use this repository, a DeepSeek API key is required, which can be found on their website. After getting the key, 

1. Get a DeepSeek API key
1. git clone this repostory
1. run in terminal "export DEEPSEEK_API_KEY={your API key}". 
2. Install the dependencies using "pip install -r requirements.txt". 
3. run 'python api_usage.py' to use the API to generate and evaluate needles. The results of the prompt are located in api_medical_needles.csv.


## Features
1. Generate needles by prompting DeepSeek to generate medical needles
2. Evaluate the same needles to detect the condition of the patient
3. Report detection rate for listed diseases (WIP)

## Technologies Used
Used OpenAI library to connect to DeepSeek API