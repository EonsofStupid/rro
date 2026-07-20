//! Zero-dependency RFC3339 timestamp parsing, for datetime filters and
//! order-preserving datetime index keys.

/// Parse an RFC3339 timestamp (`2026-07-16T02:00:00Z`,
/// `2026-07-16T04:30:00.250+02:30`, …) to epoch **milliseconds**.
/// Fractional seconds beyond milliseconds truncate. Returns `None` for
/// anything that is not a complete RFC3339 date-time.
pub fn rfc3339_to_epoch_ms(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    // Minimal complete form: YYYY-MM-DDTHH:MM:SSZ = 20 bytes.
    if b.len() < 20 {
        return None;
    }
    let num = |r: std::ops::Range<usize>| -> Option<i64> {
        let part = b.get(r)?;
        if part.iter().any(|c| !c.is_ascii_digit()) {
            return None;
        }
        std::str::from_utf8(part).ok()?.parse().ok()
    };
    let sep = |i: usize, c: u8| b.get(i) == Some(&c);

    let year = num(0..4)?;
    if !sep(4, b'-') {
        return None;
    }
    let month = num(5..7)?;
    if !sep(7, b'-') {
        return None;
    }
    let day = num(8..10)?;
    if !(b.get(10) == Some(&b'T') || b.get(10) == Some(&b't') || b.get(10) == Some(&b' ')) {
        return None;
    }
    let hour = num(11..13)?;
    if !sep(13, b':') {
        return None;
    }
    let minute = num(14..16)?;
    if !sep(16, b':') {
        return None;
    }
    let second = num(17..19)?;
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }

    // Optional fractional seconds.
    let mut i = 19;
    let mut millis: i64 = 0;
    if b.get(i) == Some(&b'.') {
        i += 1;
        let start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == start {
            return None;
        }
        let frac = &s[start..i];
        let padded = format!("{frac:0<3}");
        millis = padded[..3].parse().ok()?;
    }

    // Offset: Z or ±HH:MM.
    let offset_min: i64 = match b.get(i) {
        Some(&b'Z') | Some(&b'z') if i + 1 == b.len() => 0,
        Some(&sign @ (b'+' | b'-')) => {
            if b.len() != i + 6 || b[i + 3] != b':' {
                return None;
            }
            let oh = num(i + 1..i + 3)?;
            let om = num(i + 4..i + 6)?;
            if oh > 23 || om > 59 {
                return None;
            }
            let total = oh * 60 + om;
            if sign == b'-' {
                -total
            } else {
                total
            }
        }
        _ => return None,
    };

    // Days from civil (Howard Hinnant's algorithm): proleptic Gregorian.
    let (y, m, d) = (year, month, day);
    let y_adj = if m <= 2 { y - 1 } else { y };
    let era = if y_adj >= 0 { y_adj } else { y_adj - 399 } / 400;
    let yoe = y_adj - era * 400; // [0, 399]
    let mp = (m + 9) % 12; // Mar=0 … Feb=11
    let doy = (153 * mp + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    let days = era * 146_097 + doe - 719_468;

    let secs = days * 86_400 + hour * 3_600 + minute * 60 + second - offset_min * 60;
    Some(secs * 1_000 + millis)
}

/// Does `s` look like a canonical UUID (8-4-4-4-12 hex)? Returns the 16
/// raw bytes if so.
pub fn parse_uuid_bytes(s: &str) -> Option<[u8; 16]> {
    let b = s.as_bytes();
    if b.len() != 36 || b[8] != b'-' || b[13] != b'-' || b[18] != b'-' || b[23] != b'-' {
        return None;
    }
    let mut out = [0u8; 16];
    let mut oi = 0;
    let mut hi: Option<u8> = None;
    for (i, &c) in b.iter().enumerate() {
        if matches!(i, 8 | 13 | 18 | 23) {
            continue;
        }
        let nibble = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => return None,
        };
        match hi.take() {
            None => hi = Some(nibble),
            Some(h) => {
                out[oi] = (h << 4) | nibble;
                oi += 1;
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_known_points() {
        // The epoch itself.
        assert_eq!(rfc3339_to_epoch_ms("1970-01-01T00:00:00Z"), Some(0));
        // One day, one hour, one minute, one second, one milli.
        assert_eq!(
            rfc3339_to_epoch_ms("1970-01-02T01:01:01.001Z"),
            Some(90_061_001)
        );
        // A modern timestamp (verified: 2026-07-16T00:00:00Z).
        assert_eq!(
            rfc3339_to_epoch_ms("2026-07-16T00:00:00Z"),
            Some(1_784_160_000_000)
        );
        // Leap day.
        assert_eq!(
            rfc3339_to_epoch_ms("2024-02-29T00:00:00Z"),
            Some(1_709_164_800_000)
        );
        // Offsets shift the instant: 02:00+02:00 == 00:00Z.
        assert_eq!(
            rfc3339_to_epoch_ms("2026-07-16T02:00:00+02:00"),
            rfc3339_to_epoch_ms("2026-07-16T00:00:00Z")
        );
        assert_eq!(
            rfc3339_to_epoch_ms("2026-07-15T22:00:00-02:00"),
            rfc3339_to_epoch_ms("2026-07-16T00:00:00Z")
        );
        // Ordering follows time.
        let a = rfc3339_to_epoch_ms("2026-07-16T00:00:00Z").unwrap();
        let b = rfc3339_to_epoch_ms("2026-07-16T00:00:00.500Z").unwrap();
        let c = rfc3339_to_epoch_ms("2026-07-16T00:00:01Z").unwrap();
        assert!(a < b && b < c);
        // Junk is rejected.
        for bad in [
            "2026-07-16",
            "2026-07-16T00:00Z",
            "not a date",
            "2026-13-01T00:00:00Z",
            "2026-07-16T25:00:00Z",
            "2026-07-16T00:00:00",
            "2026-07-16T00:00:00+2:00",
        ] {
            assert_eq!(rfc3339_to_epoch_ms(bad), None, "{bad} must not parse");
        }
    }

    #[test]
    fn uuid_bytes() {
        let u = parse_uuid_bytes("550e8400-e29b-41d4-a716-446655440000").unwrap();
        assert_eq!(u[0], 0x55);
        assert_eq!(u[15], 0x00);
        assert_eq!(
            parse_uuid_bytes("550E8400-E29B-41D4-A716-446655440000"),
            Some(u),
            "case-insensitive"
        );
        assert!(parse_uuid_bytes("550e8400e29b41d4a716446655440000").is_none());
        assert!(parse_uuid_bytes("550e8400-e29b-41d4-a716-44665544000g").is_none());
    }
}
