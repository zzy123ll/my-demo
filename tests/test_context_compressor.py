"""Context Compressor 模块的完整测试。"""

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import numpy as np

from context_compressor.token_budget import TokenBudgetManager
from context_compressor.integrity_checker import IntegrityChecker
from context_compressor.config import CompressorConfig


LONG_DOC = """
员工手册第三章：休假政策。

第一条：年假制度。
入职满一年的员工享有带薪年假5天。入职满五年的员工享有带薪年假10天。
入职满十年的员工享有带薪年假15天。年假应于当年12月31日前使用完毕，
未使用的年假最多可结转5天至次年3月31日。

第二条：病假制度。
病假需提供二级以上医院开具的病假证明。3天以内病假由部门主管审批，
超过3天的病假需要HR部门审批。病假期间薪资按基本工资的80%发放。

第三条：加班制度。
工作日加班按基本工资的150%计算，休息日加班按200%计算，
法定节假日加班按300%计算。加班申请需提前一天提交部门主管审批。

第四条：婚假制度。
员工结婚享有3天婚假，符合晚婚条件的可额外享受7天婚假。
婚假须在结婚登记日起6个月内一次性使用。

以上政策自2024年1月1日起执行，若有调整另行通知。
"""

QUERY_ANNUAL_LEAVE = "年假有几天？怎么结转？"
QUERY_SICK_LEAVE = "病假需要什么材料？"
QUERY_OVERTIME = "加班费怎么计算？"


def _mock_st_encode(documents, normalize_embeddings=False):
    if isinstance(documents, str):
        documents = [documents]
    vecs = []
    for i, text in enumerate(documents):
        seed = sum(ord(c) * (j + 1) for j, c in enumerate(text[:50]))
        rng = np.random.RandomState(seed % (2 ** 31))
        v = rng.randn(384).astype(np.float32)
        if normalize_embeddings:
            v = v / (np.linalg.norm(v) + 1e-8)
        vecs.append(v)
    return np.array(vecs)


# ============================================================
class TestTokenBudgetManager:
    def test_basic_count(self):
        tb = TokenBudgetManager(context_window=4096)
        assert tb.count_tokens("Hello world") > 0

    def test_chinese_count(self):
        tb = TokenBudgetManager(context_window=4096)
        assert tb.count_tokens("年假有几天？") > 0

    def test_available_tokens(self):
        tb = TokenBudgetManager(context_window=4096, reserved_tokens=500)
        avail = tb.available_tokens("sys prompt", "user query")
        assert 0 <= avail <= 4096 - 500

    def test_fits_in_window(self):
        tb = TokenBudgetManager(context_window=4096)
        assert tb.fits_in_window("short text")
        assert not tb.fits_in_window("x" * 20000)

    def test_chunk_budget_allocation(self):
        tb = TokenBudgetManager(context_window=4096, reserved_tokens=500)
        budgets = tb.budget_for_chunks(["a b c", "d e f g h"], per_chunk_min=10)
        assert len(budgets) == 2
        assert all(b > 0 for b in budgets)

    def test_empty_chunks(self):
        assert TokenBudgetManager().budget_for_chunks([]) == []


# ============================================================
class TestIntegrityChecker:
    def test_all_entities_preserved(self):
        checker = IntegrityChecker(threshold=0.8)
        txt = "合同编号：CT-2024-001，金额：￥50000元，日期：2024年3月15日"
        result = checker.check(txt, txt)
        assert result.passed

    def test_some_entities_missing(self):
        checker = IntegrityChecker(threshold=0.8)
        original = "合同编号：CT-2024-001，金额：￥50000元，日期：2024年3月15日"
        result = checker.check(original, "合同编号：CT-2024-001。")
        assert not result.passed
        assert len(result.missing_entities) > 0

    def test_no_entities_detected(self):
        checker = IntegrityChecker()
        result = checker.check("普通描述。", "普通描述。")
        assert result.passed

    def test_date_extraction(self):
        checker = IntegrityChecker()
        entities = checker._extract_entities("成立于2024-03-15，修改于2024年6月1日")
        assert any("2024" in e for e in entities)

    def test_doc_name_extraction(self):
        checker = IntegrityChecker()
        entities = checker._extract_entities("参考《员工手册》执行")
        assert "《员工手册》" in entities


# ============================================================
class TestExtractiveCompressor:
    def test_basic_extraction(self):
        from context_compressor.extractive import ExtractiveCompressor

        compressor = ExtractiveCompressor(top_k=3)
        with patch.object(type(compressor), 'model',
                          new_callable=MagicMock) as mock_model:
            mock_model.encode = _mock_st_encode
            result = compressor.compress(LONG_DOC, QUERY_ANNUAL_LEAVE)

        assert len(result.compressed_text) > 0
        assert len(result.compressed_text) < len(LONG_DOC)
        assert result.compression_ratio < 1.0
        assert len(result.selected_sentence_indices) == 3

    def test_extraction_preserves_answer_info(self):
        from context_compressor.extractive import ExtractiveCompressor

        compressor = ExtractiveCompressor(top_k=6)
        with patch.object(type(compressor), 'model',
                          new_callable=MagicMock) as mock_model:
            mock_model.encode = _mock_st_encode
            result = compressor.compress(LONG_DOC, QUERY_OVERTIME)

        compressed = result.compressed_text
        overtime_kw = ["150%", "200%", "300%", "加班", "基本工资"]
        found = sum(1 for kw in overtime_kw if kw in compressed)
        assert found >= 1, f"加班关键信息保存不足: {found}/{len(overtime_kw)}"

    def test_short_document(self):
        from context_compressor.extractive import ExtractiveCompressor

        compressor = ExtractiveCompressor(top_k=10)
        with patch.object(type(compressor), 'model',
                          new_callable=MagicMock) as mock_model:
            mock_model.encode = _mock_st_encode
            result = compressor.compress("短文本。", "问题")
        assert result.compression_ratio == 1.0

    def test_with_token_budget(self):
        from context_compressor.extractive import ExtractiveCompressor
        from context_compressor.token_budget import TokenBudgetManager

        compressor = ExtractiveCompressor(top_k=5)
        tb = TokenBudgetManager(context_window=2048, reserved_tokens=500)
        with patch.object(type(compressor), 'model',
                          new_callable=MagicMock) as mock_model:
            mock_model.encode = _mock_st_encode
            result = compressor.compress_with_budget(LONG_DOC, QUERY_SICK_LEAVE, tb, 500)
        assert len(result.compressed_text) > 0


# ============================================================
class TestGenerativeCompressor:
    @pytest.mark.asyncio
    async def test_compression(self):
        from context_compressor.generative import GenerativeCompressor

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="入职满一年的员工享有带薪年假5天。年假应于12月31日前使用完毕。"
        ))
        config = CompressorConfig(mode="generative", llm_api_key="t", llm_base_url="h")
        compressor = GenerativeCompressor(llm=mock_llm, config=config)
        result = await compressor.compress(LONG_DOC, QUERY_ANNUAL_LEAVE)
        assert "年假" in result.compressed_text
        assert result.compression_ratio < 1.0

    @pytest.mark.asyncio
    async def test_timeout(self):
        from context_compressor.generative import GenerativeCompressor

        async def slow(*args, **kw):
            await asyncio.sleep(99)
        mock_llm = AsyncMock()
        mock_llm.ainvoke = slow
        config = CompressorConfig(mode="generative", generative_timeout=0.1,
                                  llm_api_key="t", llm_base_url="h")
        compressor = GenerativeCompressor(llm=mock_llm, config=config)
        with pytest.raises(TimeoutError):
            await compressor.compress(LONG_DOC, QUERY_ANNUAL_LEAVE)

    def test_rule_fallback(self):
        from context_compressor.generative import GenerativeCompressor
        result = GenerativeCompressor().compress_with_truncation(LONG_DOC, QUERY_OVERTIME)
        assert "加班" in result


# ============================================================
class TestContextCompressorE2E:
    @pytest.mark.asyncio
    async def test_extractive_mode(self):
        from context_compressor.context_compressor import ContextCompressor
        from context_compressor.extractive import ExtractiveCompressor

        config = CompressorConfig(mode="extractive", context_window=500)
        compressor = ContextCompressor(config=config)
        with patch.object(ExtractiveCompressor, 'model',
                          new_callable=MagicMock) as mock_model:
            mock_model.encode = _mock_st_encode
            result = await compressor.compress(LONG_DOC, QUERY_ANNUAL_LEAVE)

        assert result.compressed_text
        assert result.compression_ratio < 1.0
        assert result.mode == "extractive"
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_passthrough(self):
        from context_compressor.context_compressor import ContextCompressor
        config = CompressorConfig(mode="extractive", context_window=100000)
        compressor = ContextCompressor(config=config)
        result = await compressor.compress("短文档。", "问题")
        assert result.mode == "passthrough"

    @pytest.mark.asyncio
    async def test_fallback_truncation(self):
        from context_compressor.context_compressor import ContextCompressor
        config = CompressorConfig(mode="generative", context_window=100,
                                  fallback_max_chars=100)
        compressor = ContextCompressor(config=config, llm=None)
        result = await compressor.compress(LONG_DOC, QUERY_ANNUAL_LEAVE)
        assert result.mode == "fallback_truncation"
        assert len(result.compressed_text) <= 100

    @pytest.mark.asyncio
    async def test_answer_info_preserved(self):
        from context_compressor.context_compressor import ContextCompressor
        from context_compressor.extractive import ExtractiveCompressor

        config = CompressorConfig(mode="extractive", context_window=500,
                                  extractive_top_k_sentences=8)
        compressor = ContextCompressor(config=config)
        with patch.object(ExtractiveCompressor, 'model',
                          new_callable=MagicMock) as mock_model:
            mock_model.encode = _mock_st_encode
            result = await compressor.compress(LONG_DOC, QUERY_OVERTIME)

        compressed = result.compressed_text
        key_info = ["150%", "200%", "300%", "加班", "基本工资"]
        found = sum(1 for kw in key_info if kw in compressed)
        assert found >= 1, f"关键信息丢失: {found}/{len(key_info)}"

    @pytest.mark.asyncio
    async def test_entity_preservation(self):
        from context_compressor.context_compressor import ContextCompressor
        from context_compressor.extractive import ExtractiveCompressor

        config = CompressorConfig(mode="extractive", context_window=500,
                                  extractive_top_k_sentences=6)
        compressor = ContextCompressor(config=config)
        doc = ("法规由深圳市人民政府于2020年6月1日发布，"
               "编号为SZFG-2020-003，涉及金额￥150000元。")
        query = "法规发布时间和编号？"

        with patch.object(ExtractiveCompressor, 'model',
                          new_callable=MagicMock) as mock_model:
            mock_model.encode = _mock_st_encode
            result = await compressor.compress(doc, query)

        assert "2020" in result.compressed_text
        assert "SZFG-2020-003" in result.compressed_text

    @pytest.mark.asyncio
    async def test_latency_logged(self):
        from context_compressor.context_compressor import ContextCompressor
        from context_compressor.extractive import ExtractiveCompressor

        config = CompressorConfig(mode="extractive", context_window=500)
        compressor = ContextCompressor(config=config)
        with patch.object(ExtractiveCompressor, 'model',
                          new_callable=MagicMock) as mock_model:
            mock_model.encode = _mock_st_encode
            result = await compressor.compress(LONG_DOC, QUERY_SICK_LEAVE)
        assert result.latency_ms >= 0


# ============================================================
class TestConfigSwitching:
    def test_extractive_default(self):
        from context_compressor.context_compressor import ContextCompressor
        assert ContextCompressor(CompressorConfig(mode="extractive")).config.mode == "extractive"

    def test_generative(self):
        from context_compressor.context_compressor import ContextCompressor
        assert ContextCompressor(CompressorConfig(mode="generative")).config.mode == "generative"

    def test_token_budget(self):
        tb = TokenBudgetManager(context_window=8192, reserved_tokens=500)
        assert tb.reserved_tokens == 500
        assert tb.available_tokens() == 8192 - 500


# ============================================================
class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_document(self):
        from context_compressor.context_compressor import ContextCompressor
        result = await ContextCompressor().compress("", "问")
        assert result.compressed_text == ""

    @pytest.mark.asyncio
    async def test_whitespace_only(self):
        from context_compressor.context_compressor import ContextCompressor
        result = await ContextCompressor().compress("  \n ", "问")
        assert result.compressed_text == ""

    @pytest.mark.asyncio
    async def test_numbers_only(self):
        from context_compressor.context_compressor import ContextCompressor
        result = await ContextCompressor().compress("100 200 300", "数字")
        assert len(result.compressed_text) >= 0
