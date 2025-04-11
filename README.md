<H3> Name:ANANDA RAKSHAN K V </H3>
<H3>Register No:212223230014</H3>
<H3> Experiment 1</H3>
<H3>DATE:11/04/2025</H3>
<H1 ALIGN=CENTER> Implementation of Bayesian Networks</H1>
<h3>Aim :</h3>
    To create a bayesian Network for the given dataset in Python
<h3>Algorithm:</h3>
<h4>Step 1:</h4>
Import necessary libraries: pandas, networkx, matplotlib.pyplot, Bbn, Edge, EdgeType, BbnNode, Variable, EvidenceBuilder, InferenceController<br/>
<h4>Step 2:</h4>Set pandas options to display more columns<br/>
<h4>Step 3:</h4>Read in weather data from a CSV file using pandas<br/>
<h4>Step 4:</h4>Remove records where the target variable RainTomorrow has missing values<br/>
<h4>Step 5:</h4>Fill in missing values in other columns with the column mean<br/>
<h4>Step 6:</h4>Create bands for variables that will be used in the model (Humidity9amCat, Humidity3pmCat, and WindGustSpeedCat)<br/>
<h4>Step 7:</h4>Define a function to calculate probability distributions, which go into the Bayesian Belief Network (BBN)<br/>
<h4>Step 8:</h4>Create BbnNode objects for Humidity9amCat, Humidity3pmCat, WindGustSpeedCat, and RainTomorrow, using the probs() function to calculate their probabilities<br/>
<h4>Step 9:</h4>Create a Bbn object and add the BbnNode objects to it, along with edges between the nodes<br/>
<h4>Step 10:</h4>Convert the BBN to a join tree using the InferenceController<br/>
<h4>Step 11:</h4>Set node positions for the graph<br/>
<h4>Step 12:</h4>Set options for the graph appearance<br/>
<h4>Step 13:</h4>Generate the graph using networkx<br/>
<h4>Step 14:</h4>Update margins and display the graph using matplotlib.pyplot<br/>

## Program:
```py
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
pd.options.display.max_columns=50
import pandas as pd
df = pd.read_csv('weatherAUS.csv', encoding='utf-8')
df = df[pd.isnull(df['RainTomorrow']) == False]
numeric_df = df.select_dtypes(include=['number'])
df[numeric_df.columns] = df[numeric_df.columns].fillna(numeric_df.mean())
df['WindGustSpeedCat'] = df['WindGustSpeed'].apply(lambda x: '0.<=40' if x <= 40 else '1.40-50' if 40 < x <= 50 else '2.>50')
df['Humidity9amCat'] = df['Humidity9am'].apply(lambda x: '1.>60' if x > 60 else '0.<60')
df['Humidity3pmCat'] = df['Humidity3pm'].apply(lambda x: '1.>60' if x > 60 else '0.<=60')
def probs(data,child,parent1=None ,parent2=None):
  if parent1==None:
    prob=pd.crosstab(data[child],'Empty',margins=False,normalize='columns').sort_index().to_numpy().reshape(-1).tolist()
  elif parent1!=None:
    if parent2==None:
      prob=pd.crosstab(data[parent1],data[child],margins=False,normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else:
      prob=pd.crosstab([data[parent1],data[parent2]],data[child],margins=False,normalize='index').sort_index().to_numpy().reshape(-1).tolist()
  else:
    print("Error in Probability Frequency Calculations")
  return prob
H9am=BbnNode(Variable(0,'H9am',['<=60','>60']),probs(df,child='Humidity9amCat'))
H3pm=BbnNode(Variable(1,'H3pm',['<=60','>60']),probs(df,child='Humidity3pmCat',parent1='Humidity9amCat'))
W=BbnNode(Variable(2,'W',['<=40','40-50','>50']),probs(df,child='WindGustSpeedCat'))
RT=BbnNode(Variable(3,'RT',['No','Yes']),probs(df,child='RainTomorrow',parent1='Humidity3pmCat',parent2='WindGustSpeedCat'))
bbn=Bbn()\
.add_node(H9am)\
.add_node(H3pm)\
.add_node(W)\
.add_node(RT)\
.add_edge(Edge(H9am,H3pm,EdgeType.DIRECTED))\
.add_edge(Edge(H3pm,RT,EdgeType.DIRECTED))\
.add_edge(Edge(W,RT,EdgeType.DIRECTED))
join_tree=InferenceController.apply(bbn)
pos={0:(-1,2),1:(-1,0.5),2:(1,0.5),3:(0,-1)}
options={
    "font_size":16,
    "node_size":4000,
    "node_color":"red",
    "edgecolors":"blue",
    "edge_color":"green",
    "linewidths":5,
    "width":5,
}
n,d=bbn.to_nx_graph()
nx.draw(n,with_labels=True,labels=d,pos=pos,**options)
ax=plt.gca()
ax.margins(0.10)
plt.axis("off")
plt.show()
```
## Output:
![alt text](<Screenshot 2025-03-25 092514.png>)
## Result:
   Thus a Bayesian Network is generated using Python

