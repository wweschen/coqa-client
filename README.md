# A React-Django web site built to be a testing tool for my CoQA Tensorflow model served with a TF serving docker container.

This is a testing and demo tool for my TF2 model on CoQA data.  It can let you post a story (limited to 8000 characters), then start asking questions about the story.  My model is not very good yet, but it can answer 70% questions correctly(hopefully), well it depends on how you ask about it too.

For a data analysis of CoQA dataset, please see my [notebook](https://github.com/wweschen/Capstone/blob/master/CoQA%20dataset%20analysis.ipynb)

See it in action [here](http://wweschen.ngrok.io), if my server stays up. In case the site is not up, see the following screen shot to get a feel how it works.

![Alt text](./ScreenShot.png?raw=true "screen shot for web page")


## Quick Start
```diff
# Install dependencies
npm install

# Serve API on localhost:8000
python client/manage.py runserver

# Run webpack (from root)
npm run dev

# Build for production
npm run build

```
