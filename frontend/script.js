let sendButton = document.getElementsByClassName('send-button')[0];

console.log(sendButton)

sendButton.addEventListener("click", sendMessage)

const messages_box = document.getElementsByClassName("messages")[0]

/**
 * Sending the message to API with the user's input and then receiving the answer from backend
 */
function sendMessage() {

    let message = document.getElementsByClassName('input_user')[0].value;
    let user_box = document.createElement('div');

    user_box.classList.add("user_box");
    document.getElementsByClassName('input_user')[0].value = ''
    user_box.innerHTML = `<p>${message}</p>`;
    messages_box.append(user_box);

    $.ajax({
        url: '/get-answer',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({message: message}),
        success: function (data) {
            console.log(data)
            var message_box = document.createElement('div');
            message_box.classList.add("message_box")
            message_box.innerHTML = `<p>${data}</p>`;
            messages_box.append(message_box);
            console.log(":))")
        }
    });
}


let input_box = document.getElementsByClassName('input_user')[0];

input_box.addEventListener("keydown", sendMessageEnter);

/**
 * Sending the message if a user presses enter
 * @param event action by a user
 */
function sendMessageEnter(event){
        if (event.key === "Enter") {
            event.preventDefault();
            sendMessage();
        }
}