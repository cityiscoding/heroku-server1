<?php
/*
Plugin Name: Chatbox Integration
Description: Integrate a chatbot with FastAPI into WordPress.
Version: 1.0
Author: Your Name
*/

function chatbot_integration_scripts() {
    // In ra URL để kiểm tra
    echo plugin_dir_url(__FILE__) . 'chatbox.js';
    // Đảm bảo đường dẫn tới chatbot.js là chính xác
    wp_enqueue_script('chatbot-js', plugin_dir_url(__FILE__) . 'chatbox.js', array(), null, true);
}
add_action('wp_enqueue_scripts', 'chatbot_integration_scripts');

function chatbot_integration_shortcode() {
    return '
    <div id="chat-container">
        <div id="chat-log" style="border: 1px solid #ccc; padding: 10px; width: 300px; height: 400px; overflow-y: scroll;"></div>
        <form id="chat-form">
            <input type="text" id="chat-input" style="width: 200px;" required />
            <button type="submit">Gửi</button>
        </form>
    </div>';
}
add_shortcode('chatbot_integration', 'chatbot_integration_shortcode');
?>