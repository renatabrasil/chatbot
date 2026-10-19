from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.orchestrator import Orchestrator
from core.types import Message
from providers.bedrock_provider import BedrockProvider


def tem_evento(result, nome):
    return any(
        evento.event == nome
        for evento in result.trace
    )

def executar_eval():
    provider = BedrockProvider(
        model_id="amazon.nova-lite-v1:0",
        region_name="us-east-1"
    )

    orch = Orchestrator(provider)

    pedido_invalido = (
        "Busque voos de GRU para LIS, "
        "ida em 2027-04-29 e volta em 2027-04-10, "
        "para 2 adultos."
    )

    primeira = orch.run(
        messages=[],
        user_text=pedido_invalido,
        modo_voos=True,
    )

    assert tem_evento(
        primeira, "flight_search_validation_error"
    ), "Deveria detectar datas inválidas."

    assert not tem_evento(
        primeira, "tool_call"
    ), "Não deveria chamar a ferramenta com datas inválidas."

    historico = [
        Message(role="user", content=pedido_invalido),
        Message(role="assistant", content=primeira.answer),
    ]


    segunda = orch.run(
        messages=historico,
        user_text=(
            "Corrigindo: ida em 2027-04-10 "
            "e volta em 2027-04-29. Agora para 3 adultos."
        ),
        modo_voos=True,
    )

    chamadas = [
        evento
        for evento in segunda.trace
        if evento.event == "tool_call"
        and evento.data.get("tool") == "buscar_voos"
    ]

    assert len(chamadas) == 1, (
        "Deveria executar buscar_voos exatamente uma vez."
    )

    esperado = {
        "origem": "GRU",
        "destino": "LIS",
        "data_ida": "2027-04-10",
        "data_volta": "2027-04-29",
        "adultos": 3,
    }

    assert chamadas[0].data["tool_input"] == esperado, (
        "Os critérios da busca estão incorretos: "
        f"{chamadas[0].data['tool_input']}"
    )

    assert tem_evento(
        segunda, "tool_result"
    ), "Deveria registrar o resultado da ferramenta."

    print("PASSOU: datas inválidas bloqueadas e correção aplicada.")

if __name__ == "__main__":
    executar_eval()