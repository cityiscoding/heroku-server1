document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("chat-form");
    const input = document.getElementById("chat-input");
    const chatLog = document.getElementById("chat-log");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const userInput = input.value;
        appendMessage("Bạn", userInput);
        input.value = "";

        try {
            const response = await fetch("http://localhost:8000/chat/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ msg: userInput }),
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            appendMessage("Nhân viên", data.response);
        } catch (error) {
            console.error("Error:", error);
            appendMessage("Nhân viên", "Xin lỗi, đã có lỗi xảy ra.");
        }
    });

    function appendMessage(sender, message) {
        const messageElement = document.createElement("div");
        messageElement.textContent = `${sender}: ${message}`;
        chatLog.appendChild(messageElement);
    }
});
