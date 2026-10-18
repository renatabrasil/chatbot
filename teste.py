

from core.types import Message
from providers.bedrock_provider import BedrockProvider
from tools.flights_tool import BUSCAR_VOOS_TOOL_SPEC, BuscarVoosArgs, buscar_voos


provider = BedrockProvider(
    model_id="amazon.nova-lite-v1:0",
    region_name="us-east-1",
)

resposta = provider.solicitar_com_ferramentas([
    Message(
        role= "user",
        content=(
            "Busque voos de GRU para LIS, "
            "ida em 2027-04-10 e volta em 2027-04-29, "
            "para 1 adulto."
        ),
    ),
])

print(resposta["stopReason"])
print(resposta["output"]["message"])

mensagem_modelo = resposta["output"]["message"]

resultados_tools = []

for bloco in mensagem_modelo["content"]:
    if "toolUse" not in bloco:
        continue

    pedido = bloco["toolUse"]

    if pedido["name"] != "buscar_voos":
        raise ValueError(
            f"Ferramenta não permitida: {pedido['name']}"
        )

    # Valida os argumentos enviados pelo modelo.
    args = BuscarVoosArgs.model_validate(pedido["input"])

    # Executa nossa função Python.
    resultado = buscar_voos(args)

    resultados_tools.append({
        "toolResult": {
            "toolUseId": pedido['toolUseId'],
            "content": [{"json": resultado}]
        }
    })

    if not resultados_tools:
        raise RuntimeError("O modelo não pediu nenhuma ferramenta.")

    historico_bedrock = [
        {
            "role": "user",
            "content": [{
                "text": (
                    "Busque voos de GRU para LIS, "
                    "ida em 2027-04-10 e volta em 2027-04-29, "
                    "para 1 adulto."
                )
            }],
        },
        mensagem_modelo,
        {
            "role": "user",
            "content": resultados_tools,
        },
    ]

    resposta_final = provider.converse(
        messages=historico_bedrock,
        tools=[BUSCAR_VOOS_TOOL_SPEC],
        system=(
                "Responda em português. "
                "Informe claramente quando as ofertas forem simuladas. "
                "Use apenas os dados retornados pela ferramenta."
            ),
    )

    print("Motivo da parada:", resposta_final["stopReason"])

for bloco in resposta_final["output"]["message"]["content"]:
    if "text" in bloco:
        print(bloco["text"])