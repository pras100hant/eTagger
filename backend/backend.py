from flask import Flask
from flask import request
from flask import Response
from flask.json import jsonify
import time
import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances,sigmoid_kernel,rbf_kernel,euclidean_distances
from flask_cors import CORS, cross_origin
# app = Flask(__name__)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')
# model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

# model = SentenceTransformer('distilbert-base-nli-mean-tokens')
# model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# model = SentenceTransformer('roberta-base-nli-mean-tokens')
# model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

# model = SentenceTransformer('xlm-r-bert-base-nli-mean-tokens')
# model = SentenceTransformer('xlm-r-bert-base-nli-stsb-mean-tokens')

# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# model = SentenceTransformer('all-MiniLM-L6-v2')


epDescHash = {
	0: 'To enhance user experience, it is suggested to implement a dark UI color theme on devices equipped with AMOLED screens. Devices utilizing AMOLED screens should offer a dark UI color theme in order to improve user experience. Optimizing the user experience on devices with AMOLED screens can be achieved by providing a dark UI color theme.',
	1: 'Following a failed attempt to access a resource, it is recommended to introduce a delay between subsequent attempts, with the delay interval increasing over time. To reduce the impact of failed attempts to access a resource, it is advisable to gradually increase the time interval between each attempt made to access the same resource.',
	2:'It is advisable to prioritize tasks that are visible and valuable to the user, while avoiding tasks that quickly become obsolete. To optimize system performance, it is recommended to focus on executing tasks that provide visible value to the user, while minimizing the execution of tasks that quickly become obsolete.',
	3:'To maximize system performance, it is recommended to release resources or services as quickly as possible. Releasing resources or services promptly can help optimize system efficiency. To enhance system performance and resource utilization, it is advised to release resources or services as soon as they are no longer needed.',
	4: 'To optimize system performance, it is recommended to open or start resources and services only when strictly necessary. To minimize resource usage and improve system efficiency, it is advised to only open or start resources and services when they are absolutely required. Opening or starting resources and services only when necessary can help reduce resource consumption and improve overall system performance.',
	5: 'To optimize system performance, it is recommended to receive resource updates via push notifications, as opposed to actively querying resources. To minimize resource consumption and improve system efficiency, it is advised to utilize push notifications for receiving resource updates, rather than continuously querying resources.',
	6: 'To optimize energy usage, it is recommended to offer an energy-efficient mode that can sacrifice user experience for better energy consumption. Providing an energy-efficient mode that prioritizes energy consumption over user experience can help improve device battery life.',
	7: 'To optimize device behavior, it is recommended to adjust behavior according to power source and battery level, such as when the device is connected/disconnected to a power station. Adjusting device behavior based on power source and battery level can help improve performance and battery life.',
	8: 'To optimize data transmission, it is recommended to minimize data size as much as possible. Reducing the size of data being transmitted can help improve data transfer speeds and reduce network traffic. Minimizing the size of data being transmitted can help improve transfer rates, reduce network congestion, and improve overall system performance.',
	9: 'To optimize data usage, it is recommended to delay or disable heavy data connections until the device is connected to a WiFi network. Delaying or disabling heavy data connections until the device is connected to WiFi can help minimize mobile data usage and improve overall device performance.',
	10: 'To optimize system performance, it is recommended to avoid using intensive logging. Minimizing the use of logging can help reduce system resource consumption and improve overall system efficiency. To enhance system performance and resource utilization, it is advised to limit the use of logging, particularly intensive logging.',
	11: 'To optimize device performance, it is recommended to batch multiple operations together instead of frequently activating the device. To enhance system performance and minimize resource consumption, it is suggested to group together multiple tasks or operations into batches, rather than executing them individually and frequently activating the device.',
	12: 'To improve system efficiency and reduce resource consumption, it is recommended to use cache mechanisms to avoid performing unnecessary operations. Implementing cache mechanisms can help minimize the number of times that a system needs to perform operations, which can improve overall system performance and reduce resource usage.',
	13: 'To optimize system performance and improve battery life, it is recommended to increase the time interval between syncs and sensor reads as much as possible. Increasing the time between syncs and sensor reads can help reduce system resource consumption and improve overall system efficiency.',
	14:'To enhance device energy efficiency, it is advised to provide users with the option to enable or disable specific features as required.Allowing users to selectively enable or disable certain features can help minimize power usage, prolong battery life, and optimize device performance.',
	15: 'To improve user experience and transparency, it is recommended to notify the user when an app is performing a battery-intensive operation. Letting the user know if an app is conducting battery-intensive operations can help increase user awareness and provide them with greater control over their device\'s power usage.',
	16:'To optimize system performance and minimize resource consumption, it is recommended to collect or provide high accuracy data only when strictly necessary. Minimizing the collection or provision of high accuracy data can help reduce power usage and improve overall system efficiency.',
	17: 'To optimize resource usage and improve system efficiency, it is recommended to use data from low power sensors to infer whether new data needs to be collected from high power sensors. Inferring the need for new data collection from high power sensors using data from low power sensors can help minimize power consumption and improve overall system performance.',
	18: 'To optimize system performance and improve energy efficiency, it is recommended to terminate abnormal tasks and provide methods for interrupting energy-hungry operations. Killing abnormal tasks and providing ways to interrupt energy-intensive operations can help minimize resource consumption and improve overall system functionality.',
	19: 'To enhance user experience and optimize device functionality, it is recommended to enable interaction without requiring the use of the display whenever possible. Allowing for interaction without the need for display usage can help improve device accessibility and usability, while minimizing power consumption.',
	20: 'Graphics and animations are important components of user experience, but they can also be battery-intensive. It is recommended to use them in moderation to optimize device power usage. To enhance user experience and optimize device battery life, it is suggested to use graphics and animations judiciously, as excessive usage can deplete device power.',
	21: 'To optimize device performance and reduce resource consumption, it is recommended to perform tasks only when specifically requested by the user. Limiting task execution to only those that have been explicitly requested by the user can help improve overall system efficiency and minimize power usage.'

}
	

epDescHash = {
    0: 'Provide a dark UI color theme on devices with AMOLED screens.',
    1: 'Whenever an attempt to access a resource has failed, increase the interval of time waited before asking access to that same resource.',
    2: 'Avoid performing tasks that are not visible/valuable to the user and/or quickly become obsolete',
    3: 'Release resources or services as soon as possible',
    4: 'Open/start resources/services only when they are strictly necessary',
    5: 'Use push notifications to receive updates from resources, instead of actively querying resources',
    6: 'Provide an energy efficient mode in which user experience can drop for the sake of better energy usage',
    7: 'Have a different behavior when device is connected/disconnected to a power station, or has different battery levels',
    8: 'When transmitting data, reduce its size as much as possible',
    9: 'Delay or disable heavy data connections until the device is connected to a WiFi network',
    10: 'Avoid using intensive logging',
    11: 'Batch multiple operations instead of putting the device into an active state many times',
    12: 'Avoid performing unnecessary operations by using cache mechanisms',
    13: 'Increase time between syncs/sensor reads as much as possible',
    14: 'Allow users to enable/disable certain features in order to save energy',
    15: 'Let the user know if the app is doing any battery intensive operation',
    16: 'Collect or provide high accuracy data only when strictly necessary',
    17: 'Use data from low power sensors to infer whether new data needs to be collected from high power sensors',
    18: 'Kill abnormal tasks. Provide means of interrupting energy greedy operations',
    19: 'Whenever possible allow interaction without using the display',
    20: 'Graphics and animations are really important to improve user experience. However, they can also be battery intensive – use them with moderation',
    21: 'Perform tasks only when the user specifically asks'
}



list_of_eps = []
for ep in epDescHash:
  list_of_eps.append(epDescHash[ep])

ep_embeddings = model.encode(list_of_eps)#, normalize_embeddings=True) with this we need to just use util.dot_score with pytorch tensor
# print(ep_embeddings)
# print(ep_embeddings.shape)
def label_issue(issue_title):
    labels = []
    intensity = []
    start = time.time()
    cos_sim  = cosine_similarity([model.encode(issue_title['text'])], ep_embeddings) #normalize_embeddings in encode fun
    # cos_sim  = manhattan_distances([model.encode(issue_title['text'])], ep_embeddings) 
    
    # cos_sim  = rbf_kernel([model.encode(issue_title['text'])], ep_embeddings) 
    # cos_sim  = euclidean_distances([model.encode(issue_title['text'])], ep_embeddings) 

    end = time.time()
    time_lapse = end-start
    print("Time: ",time_lapse)
    # cs = 1/cos_sim[0]/1.5# for euclidean di75
    # cs = 1/cos_sim[0]*10 #for manhatten dist
    cs = cos_sim[0]# for rbf and cosine_sim
    #print(cos_sim.shape)
    print("Similarity metric: ",cs)
    top3eps = np.argsort(cs)[-3:]
    for p in top3eps:
        if(cs[p] >= 0.70):
            labels.append(int(p))
            intensity.append(3)
        elif(cs[p] >=0.55):
            labels.append(int(p))
            intensity.append(2)
        elif(cs[p] >=0.40):
            labels.append(int(p))
            intensity.append(1)
    return labels, intensity,time_lapse

app = Flask(__name__)
CORS(app)



@app.route("/", methods=["POST"])
# @cross_origin()
def home():
    if not 'items' in request.json:
        return "no data"
    labels = []

    time_list = []
    for issue in request.json['items']:
        # labs_inten = label_issue(issue_title=issue)
        labs, inten, time_lapse = label_issue(issue_title=issue)
        labels.append({ 
            "index": issue['index'],
            "labels": labs,
            "intensity": inten,
            })
        time_list.append(time_lapse)
    print("Avg Time:",sum(time_list)/len(time_list), "Max time:",max(time_list), "Min time:",min(time_list))
    return Response(json.dumps({
        "labels": labels
    }), mimetype='application/json')

    
if __name__ == "__main__":
    app.run(debug=True)
