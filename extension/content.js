// get all the issues div and append a tag to it. leave room for it to become conditional
var ele = document.querySelectorAll('[id^=issue_]')
c = 0
url = "127.0.0.1:5000"
issuesUrlRegex = /^https:\/\/[www.]?github.com\/[a-zA-Z0-9_]+\/[a-zA-Z0-9_]+\/issues(.*)$/

getLink = (i) => {
    return "https://tqrg.github.io/energy-patterns/#/patterns/" + epLinkHash[i]
}

function getPatternsForIssues(issues, callback) {
    // console.log(issues)
    // http request to backend
    var xhr = new XMLHttpRequest();  
    xhr.open("POST", "http://127.0.0.1:5000/");  
    xhr.setRequestHeader('Access-Control-Allow-Origin', '*');
    xhr.setRequestHeader('Content-type', 'application/json');
    xhr.onreadystatechange = ()=> { 
      // If the request completed, close the extension popup
      if (xhr.readyState == 4)
        if (xhr.status == 200){
   	        console.log("prashant singh")
            console.log(JSON.parse(xhr.responseText))//
            callback(JSON.parse(xhr.responseText))
        }
    };
    xhr.send(JSON.stringify({
        items: issues
    })); 
}

epHash = {
    0: 'Dark UI Colors',
    1: 'Dynamic Retry Delay',
    2: 'Avoid Extraneous Work',
    3: 'Race To Idle',
    4: 'Open Only When Necessary',
    5: 'Push Over Poll',
    6: 'Power Save Mode',
    7: 'Power Awareness',
    8: 'Reduce Size',
    9: 'WiFi Over Cellular',
    10: 'Supress Logs',
    11: 'Batch Operations',
    12: 'Cache',
    13: 'Decrease Rate',
    14: 'User Knows Best',
    15: 'Inform Users',
    16: 'Enough Resolution',
    17: 'Sensor Fusion',
    18: 'Kill Abnormal Tasks',
    19: 'No Screen Interaction',
    20: 'Avoid Extraneous Graphics and Animations',
    21: 'Manual Sync - On demand'
}

epLinkHash = {
    0: 'Dark_UI_Colors',
    1: 'Dynamic_Retry_Delay',
    2: 'Avoid_Extraneous_Work',
    3: 'Race-to-idle',
    4: 'Open_Only_When_Necessary',
    5: 'Push_Over_Poll',
    6: 'Power_Save_Mode',
    7: 'Power_Awareness',
    8: 'Reduce_Size',
    9: 'WiFi_Over_Cellular',
    10: 'Suppress_Logs',
    11: 'Batch_Operations',
    12: 'Cache',
    13: 'Decrease_Rate',
    14: 'User_Knows_Best',
    15: 'Inform_Users',
    16: 'Enough_resolution',
    17: 'Sensor_Fusion',
    18: 'Kill_Abnormal_Tasks',
    19: 'No_screen_interaction',
    20: 'Avoid_Extraneous_Graphics_and_Animations',
    21: 'Manual_Sync_-_On_Demand'
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

getTagsToAppend = (tags) => {
    tagElements = []
    for(let t of tags){
        tagElements.push(getTag(t, epHash[t]))
    }
    return tagElements
}

getTag = (index, tagName) => {
    element = document.createElement("div")
    element.setAttribute('class', 'tag')
    element.innerText = tagName
    link = document.createElement("a")
    link.setAttribute("href", getLink(index))
    link.setAttribute("style", "text-decoration:None")
    link.setAttribute("target", "_blank")
    link.setAttribute("title", epDescHash[index])
    link.appendChild(element)
    return link
}

async function recompute() {
    ele = document.querySelectorAll('[id^=issue_]')
    var issues = []
    for(x in ele){
        text = ele[x].innerText || ele[x].textContent
        if(x%2 != 0 && text) {
            issues.push({
                'index': x,
                'text': text
            })
        }
    }
    console.log(issues)//
    getPatternsForIssues(issues, function(taglist){
        for(let tag of taglist['labels']){
            console.log("hello"+ tag['labels'])//
            tags = getTagsToAppend(tag['labels']);
            index = tag['index']
            if(tags.length > 0){
                ul = ele[index].parentNode.appendChild(document.createElement('ul'));
                // for(let l of tags){
                //     li = ul.appendChild(document.createElement('li'))
                //     li.appendChild(l)
                //     li.style.float = 'left'
                //     li.style.margin = '2px'
                //     li.style.listStyleType = 'None'
                // }

                for(let i in tags){
                    li = ul.appendChild(document.createElement('li'))
                    li.appendChild(tags[i])
                    li.style.float = 'left'
                    li.style.margin = '2px'
                    li.style.listStyleType = 'None'
                    inten = tag['intensity'][i]

                    console.log(inten)
                    let color = ''
                    if(inten==3)
                        color = 'red'
                    else if(inten==2)
                        color = 'yellow'
                    else if(inten==1)
                        color = 'greenyellow'
                    li.style.background = color
                    li.style.borderRadius = '12px'
                    console.log(color)

                }
            }  
            
        }

    })


}


chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
      // listen for messages sent from background.js
      if (request.message === 'changed') {
        console.log(request.url) // new url is now in content scripts!
        // check url here
        if(issuesUrlRegex.test(request.url)){
        
            document.querySelectorAll('.tag').forEach(e => e.remove())
            recompute()
        }
        // 
      }
  });

// Depricated API. Please change in the future when things stop working
if(window.performance.navigation.type === 1) {
    var mydiv = document.createElement('div')
    mydiv.innerHTML = 
        `<div>
            
            <div class="dashboard" style="background: #EEE;">
                
                    <h1 style="font-size: 200px;
                                font-family: sans-serif;
                                color: #0080A8;
                                text-align: center";
                                >
                                <a href="https://tqrg.github.io/energy-patterns/#/" style="text-decoration: none;" target="_blank">
                                eTagger
                                </a>
                    <h1>
                
                
            </div>


        </div>`
    document.querySelector('#repo-content-pjax-container')
            .prepend(mydiv)

    chrome.runtime.sendMessage({message: 'reloaded', url: window.location.href}, () => {
        console.log('sending reloaded message to background..')
    })
}


