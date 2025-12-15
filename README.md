# langchain-demo

This site is to identify the best tourist destinations wihtin a country and what specific locations to watch for in these specific locations. Also, this can help in determining, what souveniers can we collect from these specific locations.

In order to install the dependencies in a virtual environment, follow the below steps.

```
python3 -m venv venv
pip install -r requirements.txt
```

In order to use ollama3

```
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3
```

In order to run the script.

```
python tourist-destinations-shortlist.py
```