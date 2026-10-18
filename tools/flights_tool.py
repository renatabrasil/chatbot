from datetime import date

from pydantic import BaseModel, Field, model_validator


class BuscarVoosArgs(BaseModel):
    origem: str = Field(
        pattern=r"^[A-Z]{3}$",
        description="Código IATA do aeroporto de origem, como GRU.",
    )
    destino: str = Field(
        pattern=r"^[A-Z]{3}$",
        description="Código IATA do aeroporto de destino, como LIS.",
    )
    data_ida: date = Field(
        description="Data de ida no formato YYYY-MM-DD.",
    )
    data_volta: date = Field(
        description="Data de volta no formato YYYY-MM-DD.",
    )
    adultos: int = Field(
        default=1,
        ge=1,
        le=9,
        description="Quantidade de passageiros adultos.",
    )

    @model_validator(mode="after")
    def validar_viagem(self):
        if self.data_volta < self.data_ida:
            raise ValueError(
                "A data de volta não pode ser anterior à data de ida."
            )

        if self.origem == self.destino:
            raise ValueError(
                "Origem e destino precisam ser diferentes."
            )

        return self

def buscar_voos(args: BuscarVoosArgs) -> dict:
    return {
        "simulado": True,
        "moeda": "BRL",
        "criterios": args.model_dump(mode="json"),
        "ofertas": [
            {
                "id": "SIM-001",
                "companhia": "Companhia Exemplo A",
                "preco_total": 4200.00 * args.adultos,
                "adultos": args.adultos,
                "escalas_ida": 0,
                "escalas_volta": 0,
                "bagagem_despachada_incluida": False,
            },
            {
                "id": "SIM-002",
                "companhia": "Companhia Exemplo B",
                "preco_total": 3650.00 * args.adultos,
                "adultos": args.adultos,
                "escalas_ida": 1,
                "escalas_volta": 1,
                "bagagem_despachada_incluida": True,
            },
        ],
    }

BUSCAR_VOOS_TOOL_SPEC = {
    "toolSpec": {
        "name": "buscar_voos",
        "description": (
            "Consulta ofertas SIMULADAS de voos de ida e volta "
            "para datas exatas e passageiros adultos. "
            "Os resultados são fictícios e servem para aprendizado. "
            "Antes de chamar, peça ao usuário os campos obrigatórios "
            "que estiverem faltando. Não invente aeroportos ou datas."
        ),
        "inputSchema": {
            "json": BuscarVoosArgs.model_json_schema()
        },
    }
}