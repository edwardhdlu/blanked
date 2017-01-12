from flask import Flask, render_template, redirect, url_for, session
from flask_pymongo import PyMongo

import re
import sys
import os
import tensorflow as tf

from bson import ObjectId
from six.moves import cPickle
from random import randint, shuffle
from model import Model


app = Flask(__name__)

app.config["MONGO_DBNAME"] = "questions"
app.debug = True # enable hot reload

mongo = PyMongo(app)
puncts = [",", "/", "\\", "\"", ":", "?", "!", "."]

app.secret_key = os.urandom(24)

GENERATE_NEW_QUESTIONS = False


def cleanse(string):
	for punct in puncts:
		string = string.replace(punct, "")

	return string


with app.app_context():
	f = file("train/data/stories/input.txt")
	text = f.read()

	vocab = re.findall(r"[\w']+", text)
	freq = {}

	for word in vocab:
		if word in freq:
			freq[word] += 1
		else:
			freq[word] = 1

	stories_pre = text.split("-----")
	stories = filter(lambda x: len(x) > 100, stories_pre)

	info_file = file("train/story-info.txt")
	story_info = info_file.readlines()

	with open(os.path.join("train/save", 'config.pkl'), 'rb') as f:
		saved_args = cPickle.load(f)
	with open(os.path.join("train/save", 'words_vocab.pkl'), 'rb') as f:
		words, vocab = cPickle.load(f)
	model = Model(saved_args, True)
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		saver = tf.train.Saver(tf.all_variables())
		ckpt = tf.train.get_checkpoint_state("train/save")

		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)

			while(GENERATE_NEW_QUESTIONS):
				r1 = randint(1, len(stories) - 1)
				story = stories[r1]
				info = story_info[r1 - 1].split("::")

				title = info[0]
				author = info[1]
				info_formatted = info[0] + "<br>" + info[1]

				sentences = story.split(".") # split by other punctuation as well
				r2 = randint(0, len(sentences) - 1)
				sentence = sentences[r2].strip() + "."
				# sentence += "\'" if sentence[0] == "\'" else ""
				# sentence +=  "\"" if sentence[0] == "\"" else ""

				sentence_words = re.findall(r"[\w']+", sentence)
				lowest_freq = sys.maxsize

				if (len(sentence_words) < 1):
					continue

				answer = cleanse(sentence_words[0])

				for word in sentence_words:
					if word in freq:
						frequency = freq[word]
						if frequency < lowest_freq:
							lowest_freq = frequency
							answer = word

				sentence_guess = sentence.replace(answer, "_____", 1)
				prime = sentence_words[sentence_words.index(answer) - 1]

				args = {}
				args["save_dir"] = "save"
				args["n"] = 2
				args["prime"] = prime
				args["sample"] = 1

				options = []
				while len(options) < 10:
					sample = model.sample(sess, words, vocab, args["n"], args["prime"], args["sample"])

					if sample == None:
						break

					new = cleanse(sample.split(" ")[1])
					if new != answer and new not in options:
						options.append(new)

				if len(options) > 0:
					options.sort(key=lambda x: freq[x] if x in freq else sys.maxsize)
					options = options[:4]
					options.append(answer)
					shuffle(options)

					new_obj = { "question": sentence_guess, "answer": answer, "title": title, "author": author, "options": options }
					mongo.db.questions.insert(new_obj)
					print mongo.db.questions.find().count()

@app.route("/")
def home():

	if "correct" not in session:
		session["correct"] = 0
	if "total" not in session:
		session["total"] = 0

	if "question_id" not in session:
		questions = mongo.db.questions.find()
		r = randint(0, questions.count() - 1)
		session["question_id"] = str(questions[r]["_id"])

	question = mongo.db.questions.find_one({ "_id": ObjectId(session["question_id"]) })
	session["answer"] = question["answer"]

	return render_template("index.html", doc=question, score=[session["correct"], session["total"]])

@app.route("/answer/<value>", methods=["GET"])
def answer(value):

	session.pop("question_id")

	session["total"] += 1
	if value == session["answer"]:
		session["correct"] += 1

	return redirect(url_for("home"))
