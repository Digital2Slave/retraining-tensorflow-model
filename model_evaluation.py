import os
import sys
import tensorflow as tf


image_dir = sys.argv[1]

total = 0. 
correct = 0.

with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
						graph_def = tf.GraphDef()
						graph_def.ParseFromString(f.read())
						tf.import_graph_def(graph_def, name='')
with tf.Session() as sess:
	for root, dirs, files in os.walk(image_dir):
		for name in files:
			if(name.endswith(".txt")):
				fullpath = image_dir + name
				with open (fullpath, "r+") as fa:

					total = total + 1
					fajpg = fa.name.replace(".txt", ".jpg")
					print("Loading file: %s" % fajpg)
					gt_cat = fa.read().strip().replace("\n", "")

					image_data = tf.gfile.FastGFile(fajpg, 'rb').read()

					label_lines = [line.rstrip() for line in tf.gfile.GFile("retrained_labels.txt")]				

					
					# Feed the image_data as input to the graph and get first prediction

					softmax_tensor = sess.graph.get_tensor_by_name('final_result:0') 
					predictions = sess.run(softmax_tensor, \
						{'DecodeJpeg/contents:0': image_data})

					top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
					#print(top_k)
					highest_prei = top_k[0]
					detected_cat = label_lines[highest_prei]
					score = predictions[0][highest_prei]
					print("Detected catagory: %s, Ground truth catagory: %s, Score: %.5f" %(detected_cat, gt_cat, score))
					if detected_cat == gt_cat:
						correct = correct + 1
						
	accuracy = (correct / total) * 100
	#print("Correct=%d" % correct)
	#print("Total=%d" % total)
	print("Evaluation complete!")
	print("Accuracy on the test sets: %.4f%%." % accuracy)