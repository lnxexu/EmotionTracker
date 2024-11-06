document.getElementById('predictButton').addEventListener('click', function() {
    const comment = document.getElementById('commentInput').value;

    if (comment.trim() === '') {
        alert('Please enter a comment.');
        return;
    }

    // Call the backend API to get the prediction
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ comment })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = `Predicted Emotion: ${data.emotion}`;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerText = 'An error occurred while predicting.';
    });
});