"""Tests for DateRangeFilter in auto_anki/contexts.py."""

import pytest
from pathlib import Path
from auto_anki.contexts import DateRangeFilter


class TestDateRangeFilter:
    """Tests for the DateRangeFilter class."""

    def test_month_filter_matches(self):
        """Month filter should match files within that month."""
        f = DateRangeFilter("2025-10")
        assert f.matches(Path("2025-10-01_topic.md"))
        assert f.matches(Path("2025-10-15_conversation.md"))
        assert f.matches(Path("2025-10-31_last_day.md"))

    def test_month_filter_excludes(self):
        """Month filter should exclude files outside that month."""
        f = DateRangeFilter("2025-10")
        assert not f.matches(Path("2025-09-30_september.md"))
        assert not f.matches(Path("2025-11-01_november.md"))
        assert not f.matches(Path("2024-10-15_last_year.md"))

    def test_range_filter_inclusive_start(self):
        """Range filter should include start date."""
        f = DateRangeFilter("2025-10-01:2025-10-15")
        assert f.matches(Path("2025-10-01_first.md"))

    def test_range_filter_exclusive_end(self):
        """Range filter end date is exclusive (>= comparison)."""
        f = DateRangeFilter("2025-10-01:2025-10-15")
        # End date is exclusive
        assert not f.matches(Path("2025-10-15_last.md"))
        assert f.matches(Path("2025-10-14_within.md"))

    def test_range_filter_within(self):
        """Range filter should match dates within range."""
        f = DateRangeFilter("2025-10-01:2025-10-31")
        assert f.matches(Path("2025-10-15_middle.md"))

    def test_range_filter_outside(self):
        """Range filter should exclude dates outside range."""
        f = DateRangeFilter("2025-10-05:2025-10-15")
        assert not f.matches(Path("2025-10-04_before.md"))
        assert not f.matches(Path("2025-10-16_after.md"))

    def test_no_filter(self):
        """No filter (None) should match everything."""
        f = DateRangeFilter(None)
        assert f.matches(Path("anything.md"))
        assert f.matches(Path("2020-01-01_old.md"))
        assert f.matches(Path("no_date_here.md"))

    def test_empty_string_filter(self):
        """Empty string should behave like no filter."""
        f = DateRangeFilter("")
        assert f.matches(Path("anything.md"))

    def test_no_date_in_filename(self):
        """Files without date in name should be included (return True)."""
        f = DateRangeFilter("2025-10")
        # Based on the code: if no date found, return True
        assert f.matches(Path("no_date_here.md"))
        assert f.matches(Path("readme.md"))

    def test_date_extraction_from_path(self):
        """Should extract date from filename correctly."""
        f = DateRangeFilter("2025-10")
        # Date can be anywhere in filename
        assert f.matches(Path("prefix_2025-10-15_suffix.md"))
        assert f.matches(Path("2025-10-15.md"))

    def test_single_date_filter(self):
        """Single date (not month, not range) should work as start date."""
        f = DateRangeFilter("2025-10-15")
        # Only start_date is set, end_date is None
        # So file_date < start_date returns False
        assert not f.matches(Path("2025-10-14_before.md"))
        assert f.matches(Path("2025-10-15_exact.md"))
        assert f.matches(Path("2025-10-16_after.md"))

    def test_december_month_rollover(self):
        """December filter should handle year rollover correctly."""
        f = DateRangeFilter("2025-12")
        assert f.matches(Path("2025-12-15_december.md"))
        # End date should be 2026-01-01
        assert not f.matches(Path("2026-01-01_january.md"))

    def test_year_boundary(self):
        """Should handle year boundaries correctly."""
        f = DateRangeFilter("2024-12-15:2025-01-15")
        assert f.matches(Path("2024-12-20_december.md"))
        assert f.matches(Path("2025-01-10_january.md"))
        assert not f.matches(Path("2025-01-15_excluded.md"))

    def test_whitespace_in_range(self):
        """Should handle whitespace around range separator."""
        f = DateRangeFilter("2025-10-01 : 2025-10-15")
        assert f.matches(Path("2025-10-10_middle.md"))

    def test_start_end_date_properties(self):
        """Should correctly set start_date and end_date properties."""
        # Month format
        f1 = DateRangeFilter("2025-10")
        assert f1.start_date == "2025-10-01"
        assert f1.end_date == "2025-11-01"

        # Range format
        f2 = DateRangeFilter("2025-10-05:2025-10-20")
        assert f2.start_date == "2025-10-05"
        assert f2.end_date == "2025-10-20"

        # No filter
        f3 = DateRangeFilter(None)
        assert f3.start_date is None
        assert f3.end_date is None
