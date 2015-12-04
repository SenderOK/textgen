# -*- coding: utf-8 -*-
import textgen

T = textgen.TextGenerator(seed=43)
T.train("./corpus")
T.save_model("./model.pkl")

# T.load_model("./model.pkl") # not necessary

text = T.generate_text(10000)
fout = open("generated_text.txt", "w")
fout.write(text.encode("utf8"))
fout.close()
