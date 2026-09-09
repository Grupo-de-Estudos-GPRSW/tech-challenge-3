# CI/CD do trabalho acadêmico

## Escopo e estrutura

A automação foi acrescentada sem modificar a lógica da aplicação nem o
`requirements.txt` original. O CI testa o pipeline com substitutos para o modelo e
o vectorstore. O CD prepara e disponibiliza um pacote de código para download.
Não há implantação em servidor, avaliação automática com GPU ou treinamento nesses
workflows. Esta é uma primeira etapa de entrega, não um deploy completo.

| Arquivo | Responsabilidade |
| --- | --- |
| `.github/workflows/ci.yml` | Sintaxe, testes rápidos, verificação das dependências e evidências. |
| `.github/workflows/cd.yml` | Reutiliza o CI e empacota o mesmo commit se os testes passarem. |
| `requirements/ci.txt` | Versões diretas para a bateria rápida. |
| `requirements/gpu.txt` | Candidato de dependências para execução/avaliação e treinamento. |

## Estado atual e evidências de execução

O projeto implementa integração contínua por meio de verificações automáticas de
dependências, sintaxe e testes do pipeline sem GPU. Após a validação na branch
principal, a automação prepara e disponibiliza um pacote associado ao commit,
acompanhado de checksum SHA-256 para verificação de integridade. As execuções
bem-sucedidas e os artefatos gerados demonstram o funcionamento desse fluxo.
A implantação automática em servidor está fora do escopo implementado.

Conforme o acompanhamento das execuções informado pela equipe:

| Componente | Status |
| --- | --- |
| CI configurado e executando automaticamente | Concluído. |
| Python/dependências e verificação de sintaxe | Configurados e validados pelo CI. |
| Testes do pipeline sem GPU | Executados com sucesso na `main`. |
| Resultados dos testes como artefato | Disponíveis. |
| CD acionado na `main` e condicionado à validação | Implementado. |
| Preparação da entrega e vínculo ao commit | Implementados. |
| Checksum SHA-256 para integridade do pacote | Gerado. |
| Execuções consecutivas bem-sucedidas | Confirmadas pela equipe. |
| Deploy automático em servidor | Não implementado. |

O artefato `resultado-pipeline` registra os testes da execução, a versão do Python,
as dependências efetivas e o commit. O artefato `entrega-<SHA>` contém o pacote e sua
identificação e integridade. Para a entrega acadêmica, preserve ambos junto aos
links das execuções correspondentes e do commit/PR, conforme a seção
[Evidências para a universidade](#evidências-para-a-universidade).

Esse estado comprova o fluxo de CI e preparação da entrega no escopo informado;
não representa implantação automática nem validação do modelo real em GPU.
O arquivo de dependências GPU continua sendo um candidato a validação, e as
dependências do CI ainda não constituem um lock completo.

## Ativar e acompanhar no GitHub

1. Envie esses arquivos em uma branch e abra um pull request para `main`.
2. Em **Actions**, acompanhe o workflow **CI**. O runner usa Ubuntu 24.04 e Python 3.11.
3. Exija o check **Pipeline sem GPU** na proteção de `main`, após a primeira execução.
4. Revise e faça o merge. O workflow **CD** executará o CI novamente
   para o commit da entrega e, se aprovado, disponibilizará o pacote.
5. Na execução, baixe `resultado-pipeline` (retenção de 14 dias) e
   `entrega-<SHA>` (retenção de 30 dias). Preserve as evidências fora do Actions antes
   de expirarem, junto ao material entregue à universidade.

Os dois workflows também permitem execução manual em **Actions → Run workflow**.
O CI aceita chamadas por `workflow_call`, usado pelo CD. Em pull requests, somente
o CI é disparado. Um push em `main` executa o CI independente e o CI chamado pelo
CD: essa repetição garante que a entrega dependa explicitamente da sua validação.
Se a branch padrão for diferente, ajuste `branches` nos dois arquivos.
Nenhuma chave OpenAI, token Hugging Face ou runner próprio é necessário nesses fluxos.

## Critérios de aprovação

- Instalação das dependências concluída e `pip check` sem conflitos declarados.
- Sintaxe Python válida, sem importar os módulos de treinamento.
- `python -m eval.battery_pipeline` termina com código zero.
- O job de empacotamento depende do sucesso do job de validação.

O comando direto da bateria é intencional: `run_eval.py` não propaga todas as
falhas dos subprocessos. A bateria retorna erro quando há casos com falha ou erro,
mas os **achados** conhecidos não bloqueiam o fluxo. O resultado histórico versionado
tem 9 casos OK e 2 achados relativos ao guardrail. Passar no CI não atesta segurança
clínica, qualidade do modelo real ou funcionamento de CUDA.

## Reproduzir os testes

Use uma cópia descartável do repositório para não sobrescrever resultados históricos
nem gerar logs na cópia principal. Na raiz dessa cópia, em Linux:

```bash
python3.11 -m venv /tmp/tech-challenge-ci
source /tmp/tech-challenge-ci/bin/activate
python -m pip install -r requirements/ci.txt
python -m pip check
export EVAL_OUTPUT_DIR="$(mktemp -d)"
python -m compileall -q src webapp eval finetuning run_ui.py run_eval.py
python -m eval.battery_pipeline
```

Em PowerShell, com um ambiente de teste já criado, o comando equivalente é:

```powershell
$env:EVAL_OUTPUT_DIR = Join-Path $env:TEMP ('avaliacao-' + [guid]::NewGuid())
python -B -m eval.battery_pipeline
if ($LASTEXITCODE -ne 0) { throw 'Falha na bateria pipeline' }
```

A bateria gera `execution_log/` no diretório de execução. No Actions, resultados
novos ficam em `runner.temp`, e os JSONs históricos em `eval/results/` são preservados.

## Dependências e limites de reprodução

As versões diretas foram obtidas do ambiente local Windows/Python 3.13.12. Elas são
uma referência inicial; a instalação limpa em Linux/Python 3.11 é verificada pelo
workflow e não deve ser declarada aprovada antes dessa execução.
Os arquivos não são locks completos: dependências transitivas ainda podem variar.
O CI arquiva `pip freeze`, a versão do Python e o commit para registrar o ambiente
efetivamente usado. Fixar versões diretas não garante reprodução integral.

Após uma execução limpa aprovada, revise o `dependencias.txt` gerado e produza um
lock específico para o sistema/Python de destino, incluindo dependências transitivas
e, idealmente, hashes. Valide esse lock recriando outro ambiente limpo antes de
adotá-lo. Não copie todo o ambiente Windows para Linux.

### Preparar GPU (pendente de definição do destino)

1. Escolha Windows ou Linux, versão Python, GPU e driver.
2. Crie um ambiente isolado e instale o build PyTorch/CUDA indicado pelo
   [seletor oficial](https://docs.pytorch.org/get-started/locally/).
3. Registre o comando, índice e versão exata usados; o PyTorch local `2.14.0`,
   sozinho, não comprova que exista um build CUDA adequado ao destino.
4. Instale `python -m pip install -r requirements/gpu.txt` e execute `pip check`.
   O arquivo é **candidato**, não uma combinação GPU validada. Dependências podem
   resolver ou atualizar PyTorch: confira a versão efetiva novamente após instalar.
5. Confira `torch.__version__`, `torch.version.cuda` e `torch.cuda.is_available()`.
6. Configure `HF_TOKEN` e `OPENAI_API_KEY` somente no ambiente; disponibilize o
   MedQuAD e o cache dos modelos. Não versione segredos ou pesos.
7. Em uma cópia de teste, execute:

   ```bash
   python run_eval.py --only model e2e --quick --retriever openai --output-dir resultados-gpu
   ```

8. Exija JSONs novos com `status: ok`, `resumo.erros: 0`, retriever OpenAI efetivo
   e resultados não vazios. Baterias puladas e código de saída zero do orquestrador
   não comprovam sucesso. Faça também um teste da interface real: o E2E usa seu
   próprio carregador do modelo.
9. Execute a avaliação sem `--quick`, compare as métricas com uma referência
   aprovada e só então registre a combinação validada e suas dependências transitivas.

## Conteúdo da entrega

`entrega-<SHA>` contém `projeto.zip`, `commit.txt` e `SHA256SUMS.txt`.
O ZIP usa `git archive HEAD` com uma lista de caminhos explícita: inclui código,
notebooks, documentação, avaliações e configurações de automação versionadas.
Não inclui a `.venv`, o `.env` da raiz, logs locais ou o diretório `.git`.
Não adicione segredos, caches ou pesos aos diretórios versionados incluídos no pacote.

Os JSONs em `eval/results/` dentro do pacote são históricos. A evidência da execução
atual é o artefato **resultado-pipeline**, que deve acompanhar o pacote.
O ZIP não contém ambiente pronto, pesos ou comprovação de inferência em GPU.

Para conferir integridade em Linux: `sha256sum -c SHA256SUMS.txt`.
Em PowerShell: `Get-FileHash ./projeto.zip -Algorithm SHA256` e compare com o arquivo.

## Evoluir para implantação

Após definir um servidor de demonstração, acrescente ao CD instalação do mesmo
pacote aprovado, segredos do ambiente, armazenamento persistente da auditoria,
inicialização do serviço e verificação de `/api/status` com `state == ready`.
HTTP 200 sozinho não basta. Teste também uma consulta fictícia via SSE e preserve
release, dependências e snapshots dos modelos anteriores para rollback.
O código atual não fixa a revisão do modelo remoto; voltar apenas o commit não
garante voltar ao mesmo modelo. Treinamento/publicação de modelos ficam separados.

## Evidências para a universidade

Os arquivos estão anexados diretamente em `docs/evidencias/`, com
[inventário](evidencias/ci-cd/README.md). As capturas mostram CI aprovado
e quatro execuções consecutivas de CD aprovadas. O CD #4 exibe o commit `c0e2d87`
na `main`, consistente com `commit.txt`, e os jobs de validação e entrega aprovados.

## Referências

- [GitHub Actions: Python](https://github.com/actions/setup-python)
- [GitHub Actions: artefatos](https://github.com/actions/upload-artifact)
- [pip: instalações repetíveis](https://pip.pypa.io/en/stable/topics/repeatable-installs/)
- [PyTorch: instalação](https://docs.pytorch.org/get-started/locally/)
