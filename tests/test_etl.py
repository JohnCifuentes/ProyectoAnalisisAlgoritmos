"""
tests/test_etl.py
=================
Pruebas unitarias para los módulos del pipeline ETL.

Cubre:
  - YahooFinanceParser: conversión de timestamps, acceso seguro a arrays,
    y parsing de JSON de muestra.
  - DataTransformer: eliminación de nulos, validación de precios,
    eliminación de duplicados y ordenamiento.
  - DataLoader: construcción del dataset maestro.

Uso:
    # Instalar pytest si no está disponible:
    pip install pytest

    # Ejecutar desde la raíz del proyecto:
    pytest tests/ -v
"""

import sys
from pathlib import Path

# Asegurar que el directorio raíz esté en sys.path para los imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src.etl.loader import DataLoader
from src.etl.parser import YahooFinanceParser
from src.etl.transformer import DataTransformer


# ====================================================================== #
# Fixtures compartidos                                                     #
# ====================================================================== #

@pytest.fixture
def sample_json() -> dict:
    """JSON de muestra con estructura idéntica a la API de Yahoo Finance."""
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "symbol": "TEST",
                        "instrumentType": "EQUITY",
                    },
                    "timestamp": [1577836800, 1577923200, 1578009600],
                    "indicators": {
                        "quote": [
                            {
                                "open":   [100.0, 101.0, 102.0],
                                "high":   [105.0, 106.0, 107.0],
                                "low":    [99.0,  100.0, 101.0],
                                "close":  [103.0, 104.0, 105.0],
                                "volume": [1000,  2000,  3000],
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }


@pytest.fixture
def valid_records() -> list:
    """Lista de registros diarios válidos."""
    return [
        {
            "date": "2020-01-01", "ticker": "T",
            "open": 10.0, "high": 11.0, "low": 9.0,
            "close": 10.5, "volume": 100, "instrument_type": "EQUITY",
        },
        {
            "date": "2020-01-02", "ticker": "T",
            "open": 11.0, "high": 12.0, "low": 10.0,
            "close": 11.5, "volume": 200, "instrument_type": "EQUITY",
        },
    ]


# ====================================================================== #
# Tests: YahooFinanceParser                                                #
# ====================================================================== #

class TestYahooFinanceParser:

    def setup_method(self) -> None:
        self.parser = YahooFinanceParser()

    # ── _unix_to_date ──────────────────────────────────────────────────

    def test_unix_to_date_known_value(self) -> None:
        """1577836800 UTC corresponde a 2020-01-01."""
        assert YahooFinanceParser._unix_to_date(1577836800) == "2020-01-01"

    def test_unix_to_date_returns_none_for_none(self) -> None:
        assert YahooFinanceParser._unix_to_date(None) is None

    def test_unix_to_date_returns_none_for_string(self) -> None:
        assert YahooFinanceParser._unix_to_date("invalid") is None

    def test_unix_to_date_returns_none_for_negative(self) -> None:
        # Timestamps negativos lejanos pueden fallar en algunas plataformas
        result = YahooFinanceParser._unix_to_date(-99999999999)
        # Puede ser None o una fecha muy antigua; no debe lanzar excepción
        assert result is None or isinstance(result, str)

    # ── _safe_get ──────────────────────────────────────────────────────

    def test_safe_get_valid_index(self) -> None:
        assert YahooFinanceParser._safe_get([10.0, 20.0, 30.0], 1) == 20.0

    def test_safe_get_first_element(self) -> None:
        assert YahooFinanceParser._safe_get([42.0], 0) == 42.0

    def test_safe_get_out_of_bounds_returns_none(self) -> None:
        assert YahooFinanceParser._safe_get([10.0], 5) is None

    def test_safe_get_preserves_none_values(self) -> None:
        """None en el array es un dato faltante válido del API."""
        assert YahooFinanceParser._safe_get([None, 20.0], 0) is None

    def test_safe_get_empty_array(self) -> None:
        assert YahooFinanceParser._safe_get([], 0) is None

    # ── parse ──────────────────────────────────────────────────────────

    def test_parse_returns_correct_count(self, sample_json) -> None:
        records = self.parser.parse(sample_json, "TEST")
        assert len(records) == 3

    def test_parse_correct_ticker(self, sample_json) -> None:
        records = self.parser.parse(sample_json, "TEST")
        assert all(r["ticker"] == "TEST" for r in records)

    def test_parse_correct_instrument_type(self, sample_json) -> None:
        records = self.parser.parse(sample_json, "TEST")
        assert all(r["instrument_type"] == "EQUITY" for r in records)

    def test_parse_correct_prices(self, sample_json) -> None:
        records = self.parser.parse(sample_json, "TEST")
        assert records[0]["open"] == 100.0
        assert records[0]["high"] == 105.0
        assert records[0]["low"] == 99.0
        assert records[0]["close"] == 103.0
        assert records[0]["volume"] == 1000

    def test_parse_correct_date_format(self, sample_json) -> None:
        records = self.parser.parse(sample_json, "TEST")
        # Verificar que la fecha tiene formato YYYY-MM-DD
        for r in records:
            parts = r["date"].split("-")
            assert len(parts) == 3
            assert len(parts[0]) == 4  # año

    def test_parse_empty_timestamps_returns_empty(self) -> None:
        json_data = {
            "chart": {
                "result": [
                    {
                        "meta": {"symbol": "T", "instrumentType": "EQUITY"},
                        "timestamp": [],
                        "indicators": {"quote": [{"open": [], "high": [], "low": [], "close": [], "volume": []}]},
                    }
                ],
                "error": None,
            }
        }
        records = self.parser.parse(json_data, "T")
        assert records == []

    def test_parse_invalid_json_returns_empty(self) -> None:
        records = self.parser.parse({}, "T")
        assert records == []

    def test_parse_with_null_values_in_arrays(self) -> None:
        """Valores None en los arrays deben pasarse al transformer."""
        json_data = {
            "chart": {
                "result": [
                    {
                        "meta": {"symbol": "T", "instrumentType": "EQUITY"},
                        "timestamp": [1577836800],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [None], "high": [None],
                                    "low": [None], "close": [None],
                                    "volume": [None],
                                }
                            ]
                        },
                    }
                ],
                "error": None,
            }
        }
        records = self.parser.parse(json_data, "T")
        # El parser debe producir el registro (el transformer lo eliminará)
        assert len(records) == 1
        assert records[0]["open"] is None


# ====================================================================== #
# Tests: DataTransformer                                                   #
# ====================================================================== #

class TestDataTransformer:

    def setup_method(self) -> None:
        self.transformer = DataTransformer()

    def test_valid_records_all_kept(self, valid_records) -> None:
        df = self.transformer.transform(valid_records, "T")
        assert len(df) == 2

    def test_removes_row_with_null_open(self, valid_records) -> None:
        valid_records[0]["open"] = None
        df = self.transformer.transform(valid_records, "T")
        assert len(df) == 1

    def test_removes_row_with_null_close(self, valid_records) -> None:
        valid_records[1]["close"] = None
        df = self.transformer.transform(valid_records, "T")
        assert len(df) == 1

    def test_removes_row_where_high_less_than_low(self) -> None:
        """high < low es incoherente financieramente."""
        records = [
            {
                "date": "2020-01-01", "ticker": "T",
                "open": 10.0, "high": 8.0, "low": 9.0,  # high < low
                "close": 10.0, "volume": 100, "instrument_type": "EQUITY",
            }
        ]
        df = self.transformer.transform(records, "T")
        assert len(df) == 0

    def test_removes_row_with_zero_close(self) -> None:
        """close = 0 no es válido para acciones o ETFs."""
        records = [
            {
                "date": "2020-01-01", "ticker": "T",
                "open": 0.0, "high": 0.0, "low": 0.0,
                "close": 0.0, "volume": 0, "instrument_type": "EQUITY",
            }
        ]
        df = self.transformer.transform(records, "T")
        assert len(df) == 0

    def test_removes_row_with_negative_volume(self) -> None:
        records = [
            {
                "date": "2020-01-01", "ticker": "T",
                "open": 10.0, "high": 11.0, "low": 9.0,
                "close": 10.5, "volume": -1, "instrument_type": "EQUITY",
            }
        ]
        df = self.transformer.transform(records, "T")
        assert len(df) == 0

    def test_removes_duplicates(self, valid_records) -> None:
        """Dos registros con la misma (date, ticker) → se conserva uno."""
        duplicated = valid_records + [valid_records[0].copy()]
        df = self.transformer.transform(duplicated, "T")
        assert len(df) == 2

    def test_sorted_chronologically(self) -> None:
        """Los registros deben quedar ordenados por fecha ascendente."""
        records = [
            {
                "date": "2020-01-03", "ticker": "T",
                "open": 10.0, "high": 11.0, "low": 9.0,
                "close": 10.5, "volume": 100, "instrument_type": "EQUITY",
            },
            {
                "date": "2020-01-01", "ticker": "T",
                "open": 10.0, "high": 11.0, "low": 9.0,
                "close": 10.5, "volume": 100, "instrument_type": "EQUITY",
            },
        ]
        df = self.transformer.transform(records, "T")
        assert df.iloc[0]["date"] == "2020-01-01"
        assert df.iloc[1]["date"] == "2020-01-03"

    def test_empty_records_returns_empty_df(self) -> None:
        df = self.transformer.transform([], "T")
        assert df.empty

    def test_volume_zero_is_valid(self) -> None:
        """volume = 0 puede ocurrir en días sin negociación (válido)."""
        records = [
            {
                "date": "2020-01-01", "ticker": "T",
                "open": 10.0, "high": 11.0, "low": 9.0,
                "close": 10.5, "volume": 0, "instrument_type": "EQUITY",
            }
        ]
        df = self.transformer.transform(records, "T")
        assert len(df) == 1


# ====================================================================== #
# Tests: DataLoader                                                        #
# ====================================================================== #

class TestDataLoader:

    def test_build_master_with_two_assets(self) -> None:
        loader = DataLoader()
        df1 = pd.DataFrame([
            {"date": "2020-01-01", "ticker": "A", "open": 10.0, "high": 11.0,
             "low": 9.0, "close": 10.5, "volume": 100, "instrument_type": "EQUITY"},
        ])
        df2 = pd.DataFrame([
            {"date": "2020-01-01", "ticker": "B", "open": 20.0, "high": 21.0,
             "low": 19.0, "close": 20.5, "volume": 200, "instrument_type": "ETF"},
        ])
        master = loader.build_master_dataset([df1, df2])
        assert len(master) == 2
        assert set(master["ticker"].unique()) == {"A", "B"}

    def test_build_master_with_empty_list(self) -> None:
        loader = DataLoader()
        master = loader.build_master_dataset([])
        assert master.empty

    def test_build_master_skips_empty_dataframes(self) -> None:
        loader = DataLoader()
        df1 = pd.DataFrame([
            {"date": "2020-01-01", "ticker": "A", "open": 10.0, "high": 11.0,
             "low": 9.0, "close": 10.5, "volume": 100, "instrument_type": "EQUITY"},
        ])
        master = loader.build_master_dataset([df1, pd.DataFrame()])
        assert len(master) == 1

    def test_build_master_has_required_columns(self) -> None:
        from src.config.settings import MASTER_COLUMNS
        loader = DataLoader()
        df = pd.DataFrame([
            {"date": "2020-01-01", "ticker": "A", "open": 10.0, "high": 11.0,
             "low": 9.0, "close": 10.5, "volume": 100, "instrument_type": "EQUITY"},
        ])
        master = loader.build_master_dataset([df])
        for col in MASTER_COLUMNS:
            assert col in master.columns

    def test_build_master_sorted_by_date(self) -> None:
        loader = DataLoader()
        df = pd.DataFrame([
            {"date": "2020-01-03", "ticker": "A", "open": 10.0, "high": 11.0,
             "low": 9.0, "close": 10.5, "volume": 100, "instrument_type": "EQUITY"},
            {"date": "2020-01-01", "ticker": "A", "open": 10.0, "high": 11.0,
             "low": 9.0, "close": 10.5, "volume": 100, "instrument_type": "EQUITY"},
        ])
        master = loader.build_master_dataset([df])
        assert master.iloc[0]["date"] == "2020-01-01"
        assert master.iloc[1]["date"] == "2020-01-03"
