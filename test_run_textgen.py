# -*- coding: utf-8 -*-
import textgen

T = textgen.TextGenerator(seed=1543)
T.train("./small_corpus")
T.save_model("./test_model.pkl")

T.load_model("./test_model.pkl")  # not necessary

text = T.generate_text(300)
fout = open("test_generated_text.txt", "w")
fout.write(text.encode("utf8"))
fout.close()
