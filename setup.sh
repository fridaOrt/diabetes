mkdir -p ~/streamlit/
echo "\
[general]\n\

email = \"fridakarina2002@icloud.com\"\n\
"> ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
""> ~/.streamlit/credentials.toml
