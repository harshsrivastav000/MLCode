# Include module turicreate
#  -*- coding: utf-8 -*-
import turicreate as tc

#height = 200
#width = 356

# Define all painting images annotations with bounding box details (I am showing only 5) 
annotations = tc.SArray([
[{'label':'PuzzleBox','type':'rectangle','coordinates':{'x':177,'y':98,'width':176,'height':187}}],
[{'label':'PuzzleBox','type':'rectangle','coordinates':{'x':175,'y':98,'width':169,'height':185}}],
[{'label':'PuzzleBox','type':'rectangle','coordinates':{'x':179,'y':98,'width':182,'height':187}}],
[{'label':'PuzzleBox','type':'rectangle','coordinates':{'x':173,'y':99,'width':173,'height':188}}],
[{'label':'PuzzleBox','type':'rectangle','coordinates':{'x':173,'y':96,'width':167,'height':186}}],
[{'label':'PuzzleBox','type':'rectangle','coordinates':{'x':174,'y':99,'width':169,'height':191}}],
[{'label':'PuzzleBox','type':'rectangle','coordinates':{'x':175,'y':100,'width':173,'height':189}}],
])

#load images by providing their relative path to the folder
images = tc.SArray([
 tc.Image('images/Object0.png'),
 tc.Image('images/Object1.png'),
 tc.Image('images/Object3.png'),
 tc.Image('images/Object4.png'),
 tc.Image('images/Object5.png'),
 tc.Image('images/Object6.png'),
 tc.Image('images/Object7.png'),

])

# Merge images and annotations
data = tc.SFrame({'image': images, 'annotations': annotations})

# Make a train-test split
train_data, test_data = data.random_split(0.8)

# Create a model using Turi Create’s object detector API
model = tc.object_detector.create(train_data, max_iterations=100)

# Save the predictions to an SArray
predictions = model.predict(test_data)

# Evaluate the model and save the results into a dictionary
metrics = model.evaluate(test_data)

# Save the model for later use in Turi Create
model.save('painting.model')

# Export for use in Core ML file to the current directory
model.export_coreml('Painting.mlmodel')
