# Projeto de Chatbot Médico

Este projeto visa ajustar um modelo de IA (finetuning) e construir um pipeline LangChain para criar um chatbot médico para auxiliar médicos.

Para utilizar CUDA, é necessário o torch compilado com superte a CUDA (uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121)

## Estrutura
- `finetuning/`: Contém os scripts e notebooks de ajuste fino.
- `src/`: Contém o código-fonte para o pipeline LangChain e outros utilitários.
