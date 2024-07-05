document.getElementById('chatbot-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const question = document.getElementById('question').value;
    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: question }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        document.getElementById('response').innerText = data.answer;
    })
    .catch((error) => {
        console.error('Error:', error);
        document.getElementById('response').innerText = 'An error occurred: ' + error.message;
    });
});
