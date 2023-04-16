var url
chrome.tabs.onUpdated.addListener(
    function(tabId, changeInfo, tab) {
      // read changeInfo data and do something with it
      // like send the new url to contentscripts.js
      if (changeInfo.status == 'complete' && tab.url) {
        if(url != tab.url){
          url = tab.url
          chrome.tabs.sendMessage( tabId, {
            message: 'changed',
            url: tab.url
          })
        }
      }
    }
  );

  chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
      // listen for messages sent from background.js
      if (request.message === 'reloaded') {
          url = request.url
          console.log('detected reload on tab ' + sender.tab.id)
          chrome.tabs.sendMessage(sender.tab.id, {
            message: 'changed',
            url: request.url
          })
      }
  });