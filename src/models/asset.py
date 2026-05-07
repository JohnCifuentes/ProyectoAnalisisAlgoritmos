"""
src/models/asset.py
===================
Modelos de datos del dominio financiero.

Define las estructuras de datos que representan los conceptos
fundamentales del proyecto:
  - Asset:       Metadatos de un instrumento financiero.
  - DailyRecord: Registro OHLCV diario de un activo.

Estas clases sirven como referencia de esquema; en la práctica
el pipeline usa DataFrames de pandas para el procesamiento en lote.
"""

from dataclasses import dataclass


@dataclass
class Asset:
    """
    Representa los metadatos de un instrumento financiero del portafolio.

    Attributes:
        ticker:      Símbolo bursátil (ej. 'AAPL', 'EC').
        name:        Nombre completo del instrumento.
        asset_type:  Tipo de instrumento: 'EQUITY' o 'ETF'.
        market:      Mercado principal de cotización.
        sector:      Sector económico al que pertenece.
    """

    ticker: str
    name: str
    asset_type: str
    market: str
    sector: str

    def __str__(self) -> str:
        return f"{self.ticker} ({self.asset_type}) — {self.name}"


@dataclass
class DailyRecord:
    """
    Representa un registro diario OHLCV de un activo financiero.

    Corresponde a una fila del dataset maestro (master_dataset.csv).

    Attributes:
        date:            Fecha de negociación en formato 'YYYY-MM-DD'.
        ticker:          Símbolo del activo.
        open:            Precio de apertura.
        high:            Precio máximo del día.
        low:             Precio mínimo del día.
        close:           Precio de cierre.
        volume:          Volumen de negociación (número de acciones/unidades).
        instrument_type: Tipo de instrumento ('EQUITY' o 'ETF').
    """

    date: str
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    instrument_type: str

    def is_valid(self) -> bool:
        """
        Verifica que el registro cumple las reglas de coherencia financiera.

        Returns:
            True si el registro es financieramente válido.
        """
        return (
            self.high >= self.low
            and self.close > 0
            and self.volume >= 0
        )
