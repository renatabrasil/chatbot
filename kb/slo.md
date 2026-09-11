## SLO - Service Level Objective

SLO é a sigla para Service Level Objective (em português, Objetivo de Nível de Serviço). Trata-se de uma meta numérica e precisa que define o nível de confiabilidade ou desempenho esperado de um serviço ou sistema. 

Os SLOs são um conceito fundamental na Engenharia de Confiabilidade de Sites (SRE, na sigla em inglês), metodologia originada no Google, e servem para estabelecer, de forma clara e mensurável, o que é um serviço "bom o suficiente" para o usuário final. 

## Componentes Chave do SLO
- Métrica Quantificável: Um SLO é sempre baseado em um Indicador de Nível de Serviço (SLI), que é a métrica utilizada para medir o desempenho do serviço (por exemplo, tempo de atividade, latência, taxa de erros).
- Alvo Específico: Define o objetivo a ser alcançado. Por exemplo: "A latência da API deve ser inferior a 300ms em 99% das requisições em um período de 30 dias".
- Período de Tempo: O objetivo é medido dentro de um intervalo de tempo definido, como um mês ou uma semana. 

## Exemplo Prático

Imagine um serviço de e-commerce:
- SLI: Medição do tempo de resposta da página de checkout.
- SLO: A página de checkout deve responder em menos de 1 segundo para 99,9% dos usuários, em um período contínuo de 30 dias. 

## Diferença entre SLI, SLO e SLA
É comum confundir os termos, mas eles têm papéis distintos: 
- SLI (Service Level Indicator): O indicador, a métrica bruta que você mede (ex: "tempo de resposta médio em milissegundos").
- SLO (Service Level Objective): O objetivo ou a meta baseada no SLI (ex: "tempo de resposta abaixo de 1s em 99% do tempo").
- SLA (Service Level Agreement): O contrato formal entre o provedor do serviço e o cliente que especifica as consequências (geralmente financeiras ou contratuais) caso os SLOs não sejam atingidos. 

O SLO permite que as equipes de engenharia tenham uma margem de tolerância (conhecida como "orçamento de erro") para inovações e falhas, garantindo que o serviço atenda às expectativas do cliente sem buscar uma perfeição de 100% que muitas vezes é inviável ou cara demais. 
