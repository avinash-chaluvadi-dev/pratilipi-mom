# Hi, I'm Headliner! 👋

I am a module known for giving out headline of paragraphs of text, and I am cooool! 🧊

**What really is a Headline generation ?** : `Headline Generation is a process of generating an appropriate and most informative headline which is in line with the concise summary. It is a special type of text summarization task.`

## Here is an example 👇

**Original**
Very early yesterday morning, the United States President Donald Trump reported he and his wife First Lady Melania Trump tested positive for COVID-19. Officials said the Trumps' 14-year-old son Barron tested negative as did First Family and Senior Advisors Jared Kushner and Ivanka Trump.
Trump took to social media, posting at 12:54 am local time (0454 UTC) on Twitter, "Tonight, [Melania] and I tested positive for COVID-19. We will begin our quarantine and recovery process immediately. We will get through this TOGETHER!" Yesterday afternoon Marine One landed on the White House's South Lawn flying Trump to Walter Reed National Military Medical Center (WRNMMC) in Bethesda, Maryland.
Reports said both were showing "mild symptoms". Senior administration officials were tested as people were informed of the positive test. Senior advisor Hope Hicks had tested positive on Thursday.
Presidential physician Sean Conley issued a statement saying Trump has been given zinc, vitamin D, Pepcid and a daily Aspirin. Conley also gave a single dose of the experimental polyclonal antibodies drug from Regeneron Pharmaceuticals.
According to official statements, Trump, now operating from the WRNMMC, is to continue performing his duties as president during a 14-day quarantine. In the event of Trump becoming incapacitated, Vice President Mike Pence could take over the duties of president via the 25th Amendment of the US Constitution. The Pence family all tested negative as of yesterday and there were no changes regarding Pence's campaign events.

**Headline**
Trump and First Lady Melania Test Positive for COVID-19

## How to use me ? 💁

```javascript
from module.Headliner import model_serve

response = model_serve(test_input)
```

## Want to test me ? 🧐

No issues! check me, run the following command. Let me give you some dependency issues

```
python main.py --mode=package_test --func_test=all --file_path=path/to/file.json [optional]
```

**Here are the different options for package test**:

- _func_test_ = "all" -> runs test for all the components.
- _func_test_ = "train" -> runs the test for training component only.
- _func_test_ = "serve" -> runs the test for serving component only.
- _func_test_ = "eval" -> runs the test for evaluation component only.

#### Here is my Pluggable Component ⚓

This mode of running the package, showcase the capability to be able to plug in the train, eval and serving
component of this module into API integration or MLOPs engine.

- _mode_="train" - run the training component of the package.
- _mode_="serve" - run the serving component of the package.
- _mode_="eval" - run the evaluation component of the package.

`python main.py --mode=serve`

### Do the Parameters Setup

**Network Hyperparameters**

- **DEVICE** - "cpu" or "gpu", depending upon the hardware supported by your local machine / your choice
- **BATCH_SIZE:** Number of samples in a single batch.
- **NUM_EPOCHS:** Number of epochs for training the model.
- **LEARNING_RATE:** Tuning parameter in the optimization algorithm that determines the step size at each iteration.
- **INPUT_LENGTH:** Max length of the input texts
- **OUTPUT_LENGTH:** Max length of the output headlines
- **DIVIDE_TEXT:** Whether to divide the merged texts into smaller chunks or not ?
- **DIVIDE_N:** Number of chunks to make from the merged texts 

## Something related to installation 🔨

Dude! Install the dependencies.

```sh
pip install -r requirements.txt
```

## How to fuel me ? ⛽

Provide me with some data for training inside dataset/ folder and don't forget to change the _config.py_ for the location of the file.

## Train me How? 🚅

**You have no idea. Really? Why don't I give you a hint.**

```sh
python main.py --mode=train
```

That was more than a hint. But it's okay

`🛑 Danger Alert:  You might get an error while using the model for train/ eval / serve like this: *RuntimeError: CUDA out of memory.* This is due to large size of input data while using CUDA, you can change DEVICE in config.py to cpu to overcome this. Although this might slow the process.`

## QNA Time!!! 📣

**Q1. What kind of model is used in this package ?**
**Ans.** [T5 model](https://huggingface.co/transformers/model_doc/t5.html#t5forconditionalgeneration) (Text-To-Text Transfer Transformer) is used as a base model by me.

**Q2. What do you mean by base model ? Are you not using it directly ?**
**Ans.** Good question! I am using the T5 model as a base model. Then I have been trained on a collection of 500k articles with headings. My purpose is to create a one-line heading suitable for the given article.

**Q3. What is this T5 model ?**
**Ans.** [T5](https://huggingface.co/transformers/model_doc/t5.html) (Text-To-Text Transfer Transformer) is an encoder-decoder model and converts all NLP problems into a text-to-text format, which means the input and output are both text strings. T5 is an extremely large new neural network model that is trained on a mixture of unlabeled text (the authors’ huge new C4 collection of English web text) and labeled data from popular natural language processing tasks, then fine-tuned individually for each of the tasks that they author aim to solve. It works quite well, setting the state of the art on many of the most prominent text classification tasks for English, and several additional question-answering and summarization tasks as well. Isn't it interesting ??

**Q4. What are the different kind of format supported for input data ?**
**Ans.** Currently I support JSON and list format as an input for the API.

**Q5. What if I want to see the logs of what is happening ?**
**Ans.** That too has been taken care of. Whenever the process is completed, you can go to results/run_logs/ to see the logs.

**Q6. Is this Readme written by you ?**
**Ans.** Of course, YES. (Hiding my lying face! 😬)

## My Roadmap✈️

- Adding the retraining part

## This Guy made me 🦸‍♂️

- Amit Joshi _aka_ A.J
