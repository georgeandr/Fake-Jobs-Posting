### 
# 
# Libraries Imports
#
###
import pyspark.sql
import pyspark.sql.functions as psf
import pyspark.ml as pml
from sparknlp.annotator import BertSentenceEmbeddings, ClassifierDLApproach
from sparknlp import DocumentAssembler
import sparknlp
from sparknlp.base import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import classification_report

### 
# 
# Creating a SparkSession called Project Decentralized Systems
#spark = pyspark.sql.SparkSession.builder.appName("Project Job Posting Classification").getOrCreate()
#
###
spark = sparknlp.start(gpu = True)

###
#  
# Reading the CSV file 
#
###
df = spark.read.option("header","true") \
               .option("sep", ",") \
               .option("multiLine", "true") \
               .option("quote","\"") \
               .option("escape","\"") \
               .option("ignoreTrailingWhiteSpace", True) \
               .csv("../Project/fake_job_postings.csv")
df.printSchema()
df.show()

### 
# 
# Dropping columns except Description and Fraudulent which will be used for the classification ###
#
###
columns_of_df = df.columns
df_jd_fr = df.drop(*[c for c in columns_of_df if c not in {"description", "fraudulent"}]) ### df_jd_fr equals with dataframe with only job description and fraudulent columns
df_jd_fr.printSchema()
df_jd_fr.show(10)

###
#
# Calculating the number of NULL values in each column and also the number of instances per class
#
###
columns_of_df_jd_fr = df_jd_fr.columns
df_jd_fr.select([psf.count(psf.when(psf.col(c).isNull(), c)).alias(c) for c in columns_of_df_jd_fr]).show()
df_jd_fr.groupBy("fraudulent").agg(psf.count("fraudulent").alias("Number of rows per class")).show()

###
#
# Setting up a pipeline in order to transform the values of column "description" into vectors that later will fed into our neural network to predict their class.
# DocumentAssembler is used to define our document with input the "description" column.
#
###
documentAssembler = DocumentAssembler() \
    .setInputCol("description") \
    .setOutputCol("document")

embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L8_128", "en") \
      .setInputCols("document") \
      .setOutputCol("sentence_embeddings")

classifierDL = ClassifierDLApproach() \
    .setInputCols("sentence_embeddings") \
    .setOutputCol("predicted") \
    .setLabelColumn("label") \
    .setMaxEpochs(10) \
    .setEnableOutputLogs(True)        

pipeline = Pipeline().setStages([
    documentAssembler,
    embeddings,
    classifierDL
])

df_jd_fr = df_jd_fr.withColumn("label", df_jd_fr.fraudulent.cast("int"))
training_bert_transformed_df_jb_fr, test_bert_transformed_df_jb_fr  = df_jd_fr.randomSplit([0.8, 0.2])
training_bert_transformed_df_jb_fr.groupBy("fraudulent").agg(psf.count("fraudulent").alias("Number of rows per class in training set")).show()
test_bert_transformed_df_jb_fr.groupBy("fraudulent").agg(psf.count("fraudulent").alias("Number of rows per class in test set")).show()

model = pipeline.fit(training_bert_transformed_df_jb_fr)
preds = model.transform(test_bert_transformed_df_jb_fr)

preds_df = preds.select('fraudulent','description',"predicted.result").toPandas()
preds_df['result'] = preds_df['result'].apply(lambda x : x[0])
print(classification_report(preds_df['result'], preds_df['fraudulent']))

### Using the pyspark tokenizer with input the "description" column and the result is stored into the column "words".
#
# Then tokenizer is used to transofrm our dataframe into tokens. I am also counting the number of toknes-words per row
# and the output is stored into column "number_of_words". 
# Finally, StoptWordsRemover is used to remove the stop words of each row of column "words" and the result is stored
# into the column "filtered_tokens_words".
# To evaluate that the StopWordsRemover worked, I counted the number of filtered tokens words and the output
# is stored into the column "number_of_filtered_tokens_words" 
#
###
tokenizer = pml.feature.Tokenizer(inputCol = "description", outputCol = "words")
tokenized = tokenizer.transform(df_jd_fr)
count_words = psf.udf(lambda words: len(words), pyspark.sql.types.IntegerType())
df_jd_fr = tokenized.select("description", "fraudulent",  "words") \
           .withColumn("number_of_words", count_words(psf.col("words")))        
remover = pml.feature.StopWordsRemover(inputCol = "words", outputCol = "filtered_tokens_words")
df_jd_fr = remover.transform(df_jd_fr)
count_filtered_tokens = psf.udf(lambda filtered_tokens_words: len(filtered_tokens_words), pyspark.sql.types.IntegerType())
df_jd_fr = df_jd_fr.select("description", "fraudulent",  "words", "number_of_words", "filtered_tokens_words") \
          .withColumn("number_of_filtered_tokens_words", count_filtered_tokens(psf.col("filtered_tokens_words"))) 
df_jd_fr.show()

###
#
# Using Word2Vec function to transform the "filtered_tokens_words" to vectors and the result is stored into a new dataframe.
#
###
word2vec = pml.feature.Word2Vec(vectorSize = 100, seed = 42, maxIter = 10, inputCol = "filtered_tokens_words", outputCol = "vectorized_words")
word2vec_transformed_df_jd_fr = word2vec.fit(df_jd_fr).transform(df_jd_fr)
word2vec_transformed_df_jd_fr = word2vec_transformed_df_jd_fr.withColumn("label", word2vec_transformed_df_jd_fr.fraudulent.cast("int"))
word2vec_transformed_df_jd_fr.show()
word2vec_transformed_df_jd_fr.printSchema()

training_word2vec_transformed_df_jb_fr, test_word2vec_transformed_df_jb_fr  = word2vec_transformed_df_jd_fr.randomSplit([0.8, 0.2])
training_word2vec_transformed_df_jb_fr.groupBy("fraudulent").agg(psf.count("fraudulent").alias("Number of rows per class in training set")).show()
test_word2vec_transformed_df_jb_fr.groupBy("fraudulent").agg(psf.count("fraudulent").alias("Number of rows per class in test set")).show()

layers = [100, 5, 4, 2]
trainer = MultilayerPerceptronClassifier(featuresCol = "vectorized_words", labelCol = "label", maxIter = 10,layers = layers, blockSize = 128, seed = 1234)
model = trainer.fit(training_word2vec_transformed_df_jb_fr)

result = model.transform(test_word2vec_transformed_df_jb_fr)
preds_df = result.select("prediction", "label")
evaluator1 = MulticlassClassificationEvaluator(metricName = "accuracy")
evaluator2 = MulticlassClassificationEvaluator(metricName = "precisionByLabel")
evaluator3 = MulticlassClassificationEvaluator(metricName = "recallByLabel")
evaluator4 = MulticlassClassificationEvaluator(metricName = "f1")
print("Test set accuracy = " + str(evaluator1.evaluate(preds_df)) + "\n Test set precision =" + str(evaluator2.evaluate(preds_df)) 
+ "\n Test set recall = " + str(evaluator3.evaluate(preds_df)) + "\n Test set F1 = " + str(evaluator4.evaluate(preds_df)))