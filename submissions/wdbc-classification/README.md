# WDBC Classification using Hybrid Models

* We have use Classical - Quantum - Classical in NN model.
* To run the model, open `wdbc-classification.ipynb` and run the sections.
* We have divided the notebook into sections. We can retrain the model or load the trained model and test it with the test data.
* There was no explicit test data so we split the data with training dataset having 338 points, validation dataset having 60 and test dataset having 171
* The qunatum circuits has below structure:
      0: ─╭AngleEmbedding(M0)─╭RandomLayers(M1)─╭StronglyEntanglingLayers(M2)─┤  <Z>
      1: ─├AngleEmbedding(M0)─├RandomLayers(M1)─├StronglyEntanglingLayers(M2)─┤     
      2: ─├AngleEmbedding(M0)─├RandomLayers(M1)─├StronglyEntanglingLayers(M2)─┤     
      3: ─├AngleEmbedding(M0)─├RandomLayers(M1)─├StronglyEntanglingLayers(M2)─┤     
      4: ─├AngleEmbedding(M0)─├RandomLayers(M1)─├StronglyEntanglingLayers(M2)─┤     
      5: ─├AngleEmbedding(M0)─├RandomLayers(M1)─├StronglyEntanglingLayers(M2)─┤     
      6: ─├AngleEmbedding(M0)─├RandomLayers(M1)─├StronglyEntanglingLayers(M2)─┤     
      7: ─├AngleEmbedding(M0)─├RandomLayers(M1)─├StronglyEntanglingLayers(M2)─┤     
      8: ─├AngleEmbedding(M0)─├RandomLayers(M1)─├StronglyEntanglingLayers(M2)─┤     
      9: ─╰AngleEmbedding(M0)─╰RandomLayers(M1)─╰StronglyEntanglingLayers(M2)─┤
* The structure of classical NN has Four layers:
    1. Input : Matches the dimension of input dataset ( 30 )
    2. Hidden : It is equivalent to 20
    3. Sub Output: This has 10 nodes and is connected to qunatum
    4. Classifier: This gets results from the qunatum layer

* Since the dataset was not balanced we have used nn.BCEWithLogitsLoss with pos_weight calculated based on the dataset ration.
* The final result metrics are:
    Precision:     1.0000
    Recall:        0.9589
    F1 Score:      0.9790
    Accuracy:      0.9825
    True Positives:  70
    False Positives: 0
    True Negatives:  98
    False Negatives: 3
