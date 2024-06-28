use std::any::Any;
use std::collections::HashMap;
use std::fs::{metadata, File};
use std::io::Read;
use std::mem::size_of;
use std::str::FromStr;

pub const BLOCK_SIZE: u64 = 2880;

#[derive(Clone)]
pub enum HeaderValue {
    Stringval { v: String },
    Logical { v: bool },
    Integer { v: i64 },
    Float { v: f64 },
    Complex { re: f64, im: f64 },
    Empty,
}

pub trait TypeCompatible {
    fn is_compatible(value: &HeaderValue) -> bool;
}

impl TypeCompatible for String {
    fn is_compatible(value: &HeaderValue) -> bool {
        if let HeaderValue::Stringval { v: _ } = value {
            true
        } else {
            false
        }
    }
}

impl TypeCompatible for bool {
    fn is_compatible(value: &HeaderValue) -> bool {
        if let HeaderValue::Logical { v: _ } = value {
            true
        } else {
            false
        }
    }
}

impl TypeCompatible for i64 {
    fn is_compatible(value: &HeaderValue) -> bool {
        if let HeaderValue::Integer { v: _ } = value {
            true
        } else {
            false
        }
    }
}

impl TypeCompatible for f64 {
    fn is_compatible(value: &HeaderValue) -> bool {
        if let HeaderValue::Float { v: _ } = value {
            true
        } else {
            false
        }
    }
}

impl TryFrom<&HeaderValue> for String {
    type Error = ();
    fn try_from(value: &HeaderValue) -> Result<Self, Self::Error> {
        if let HeaderValue::Stringval { v } = value {
            Ok(v.clone())
        } else {
            Err(())
        }
    }
}

impl TryFrom<&HeaderValue> for bool {
    type Error = ();
    fn try_from(value: &HeaderValue) -> Result<Self, Self::Error> {
        if let HeaderValue::Logical { v } = value {
            Ok(v.clone())
        } else {
            Err(())
        }
    }
}

impl TryFrom<&HeaderValue> for i64 {
    type Error = ();
    fn try_from(value: &HeaderValue) -> Result<Self, Self::Error> {
        if let HeaderValue::Integer { v } = value {
            Ok(v.clone())
        } else {
            Err(())
        }
    }
}

impl TryFrom<&HeaderValue> for f64 {
    type Error = ();
    fn try_from(value: &HeaderValue) -> Result<Self, Self::Error> {
        if let HeaderValue::Float { v } = value {
            Ok(v.clone())
        } else {
            Err(())
        }
    }
}

impl HeaderValue {
    pub fn get<T>(&self) -> Option<T>
    where
        T: TypeCompatible + for<'a> TryFrom<&'a HeaderValue>,
    {
        if T::is_compatible(self) {
            if let Ok(v) = T::try_from(self) {
                Some(v)
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[derive(Clone)]
pub struct HeaderEntry {
    value: HeaderValue,
    comment: Option<String>,
    position: u32,
}

pub enum HeaderGroup {
    Single(Option<HeaderEntry>),
    Multi(Vec<HeaderEntry>),
}

pub enum HduType {
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
}

pub struct ImageHdu<T: ToHDU> {
    data_type: HduType,
    header: HashMap<String, HeaderGroup>,
    data: Option<Vec<T>>,
}

pub trait ToHDU {
    type Target;
    fn from_be(bytes: &[u8]) -> Self::Target;
    fn data_type() -> HduType;
}

impl ToHDU for u8 {
    type Target = u8;

    fn from_be(bytes: &[u8]) -> Self::Target {
        bytes[0]
    }

    fn data_type() -> HduType {
        HduType::Int8
    }
}

impl ToHDU for i16 {
    type Target = i16;

    fn from_be(bytes: &[u8]) -> i16 {
        let arr: [u8; 2] = [bytes[0], bytes[1]];
        i16::from_be_bytes(arr)
    }

    fn data_type() -> HduType {
        HduType::Int16
    }
}

impl ToHDU for i32 {
    type Target = i32;

    fn from_be(bytes: &[u8]) -> i32 {
        let arr: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
        i32::from_be_bytes(arr)
    }

    fn data_type() -> HduType {
        HduType::Int32
    }
}

impl ToHDU for i64 {
    type Target = i64;

    fn from_be(bytes: &[u8]) -> i64 {
        let arr: [u8; 8] = [
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ];
        i64::from_be_bytes(arr)
    }

    fn data_type() -> HduType {
        HduType::Int64
    }
}

impl ToHDU for f32 {
    type Target = f32;

    fn from_be(bytes: &[u8]) -> f32 {
        let arr: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
        f32::from_be_bytes(arr)
    }

    fn data_type() -> HduType {
        HduType::Float32
    }
}

impl ToHDU for f64 {
    type Target = f64;

    fn from_be(bytes: &[u8]) -> f64 {
        let arr: [u8; 8] = [
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ];
        f64::from_be_bytes(arr)
    }

    fn data_type() -> HduType {
        HduType::Float64
    }
}

pub trait Hdu {
    fn get_type(self: &Self) -> &HduType;

    fn header(self: &Self) -> &HashMap<String, HeaderGroup>;

    fn as_any(self: &Self) -> &dyn Any;
}

impl<T: ToHDU + 'static> Hdu for ImageHdu<T> {
    fn get_type(self: &Self) -> &HduType {
        &self.data_type
    }

    fn header(self: &Self) -> &HashMap<String, HeaderGroup> {
        &self.header
    }

    fn as_any(self: &Self) -> &dyn Any {
        self
    }
}

pub fn open(file_name: &str) -> Result<Vec<Box<dyn Hdu>>, String> {
    match open_file(file_name) {
        Ok(mut file) => read_file_fully(&mut file),
        Err(e) => Err(e),
    }
}

fn read_file_fully(file: &mut impl Read) -> Result<Vec<Box<dyn Hdu>>, String> {
    let mut result: Vec<Box<dyn Hdu>> = Vec::with_capacity(10);
    loop {
        let maybe_header = read_header(file, &result.is_empty());
        if maybe_header.is_err() {
            return Err(maybe_header.err().unwrap());
        }
        let maybe_header = maybe_header.unwrap();
        if let Some(header) = maybe_header {
            let maybe_data = get_data(file, &header);
            if maybe_data.is_err() {
                return Err(maybe_data.err().unwrap());
            }
            let data = maybe_data.unwrap();
            let hdu: Box<dyn Hdu> = match get_int_header(&header, "BITPIX") {
                Some(8) => Box::new(ImageHdu {
                    data_type: HduType::Int8,
                    header,
                    data,
                }),
                Some(16) => Box::new(create_hdu::<i16>(header, data)),
                Some(32) => Box::new(create_hdu::<i32>(header, data)),
                Some(64) => Box::new(create_hdu::<i64>(header, data)),
                Some(-32) => Box::new(create_hdu::<f32>(header, data)),
                Some(-64) => Box::new(create_hdu::<f64>(header, data)),
                Some(x) => {
                    return Err(format!(
                        "Unexpected BITPIX value {} in segment #{}",
                        x,
                        result.len() + 1
                    ))
                }
                None => {
                    return Err(format!(
                        "Missing BITPIX value in segment #{}",
                        result.len() + 1
                    ))
                }
            };
            result.push(hdu);
        } else {
            return Ok(result);
        }
    }
}

fn create_hdu<T>(header: HashMap<String, HeaderGroup>, data: Option<Vec<u8>>) -> ImageHdu<T>
where
    T: ToHDU<Target = T>,
{
    if data.is_none() {
        return ImageHdu {
            data_type: T::data_type(),
            header,
            data: None,
        };
    }

    let size_of_t = size_of::<T>();

    let data = data.unwrap();
    let mut res_data: Vec<T> = Vec::with_capacity(data.len() / size_of_t);
    for chunk in data.chunks_exact(size_of_t) {
        res_data.push(T::from_be(chunk))
    }
    ImageHdu {
        data_type: T::data_type(),
        header,
        data: Some(res_data),
    }
}

fn open_file(file_name: &str) -> Result<File, String> {
    return match metadata(file_name) {
        Ok(metadata) if metadata.len() % BLOCK_SIZE != 0 => {
            Err(format!("File size isn't multiple of {BLOCK_SIZE}"))
        }
        Ok(_) => match File::open(file_name) {
            Ok(file) => Ok(file),
            Err(e) => Err(format!("Error opening file: {e}")),
        },
        Err(e) => Err(format!("Error getting metadata for file: {e}")),
    };
}

fn get_data(
    file: &mut impl Read,
    header: &HashMap<String, HeaderGroup>,
) -> Result<Option<Vec<u8>>, String> {
    let data_size = get_data_size(header);
    if data_size == 0 {
        return Ok(None);
    }
    let mut buffer_size = (data_size / BLOCK_SIZE as usize) * BLOCK_SIZE as usize;
    if data_size % BLOCK_SIZE as usize != 0 {
        buffer_size += BLOCK_SIZE as usize;
    }
    let mut buffer: Vec<u8> = Vec::with_capacity(buffer_size);
    buffer.resize(buffer_size, 0);
    match read_block(file, buffer.as_mut_slice()) {
        Ok(num_read) if num_read != buffer_size => Err(format!(
            "error reading data: only read {num_read} of {buffer_size} bytes expected."
        )),
        Err(e) => Err(e),
        _ => {
            buffer.drain(data_size..buffer_size);
            buffer.shrink_to_fit();
            Ok(Some(buffer))
        }
    }
}

fn get_data_size(header: &HashMap<String, HeaderGroup>) -> usize {
    let mut raw_size = get_int_header(header, "BITPIX").unwrap().abs() / 8;
    let naxis = get_int_header(header, "NAXIS").unwrap();
    if naxis == 0 {
        return 0;
    }
    let mut nelem: i64 = 1;
    for n in 1..naxis + 1 {
        let key = format!("NAXIS{n}");
        nelem *= get_int_header(header, &key).unwrap();
    }
    if let Some(v) = get_int_header(header, "PCOUNT") {
        nelem += v;
    }
    if let Some(v) = get_int_header(header, "GCOUNT") {
        nelem *= v;
    }
    raw_size *= nelem;

    raw_size as usize
}

fn get_int_header(header: &HashMap<String, HeaderGroup>, key: &str) -> Option<i64> {
    if let Some(HeaderGroup::Single(Some(HeaderEntry { value, .. }))) = header.get(key) {
        value.get::<i64>()
    } else {
        None
    }
}

fn read_header(
    file: &mut impl Read,
    primary: &bool,
) -> Result<Option<HashMap<String, HeaderGroup>>, String> {
    let mut buffer: Vec<u8> = Vec::with_capacity(BLOCK_SIZE as usize);
    buffer.resize(BLOCK_SIZE as usize, 0);
    let mut header: HashMap<String, HeaderGroup> = HashMap::with_capacity(20);
    let mut current_pos: u32 = 0;
    loop {
        match read_header_block(file, &mut buffer, &mut header, &mut current_pos, primary) {
            Ok(is_done) if is_done => {
                if current_pos > 0 {
                    return Ok(Some(header));
                } else {
                    return Ok(None);
                }
            }
            Err(e) => {
                return Err(e);
            }
            _ => {}
        }
    }
}

fn read_header_block(
    file: &mut impl Read,
    buffer: &mut Vec<u8>,
    header: &mut HashMap<String, HeaderGroup>,
    current_pos: &mut u32,
    primary: &bool,
) -> Result<bool, String> {
    match read_block(file, buffer.as_mut_slice()) {
        Ok(bytes_read) if bytes_read > 0 => {
            match parse_header_block(buffer, header, current_pos, primary) {
                Ok(v) => Ok(v),
                Err(e) => Err(e),
            }
        }
        Ok(_) => {
            if *primary || *current_pos > 0 {
                Err(String::from("end of file before end of header"))
            } else {
                Ok(true)
            }
        }
        Err(e) => Err(e),
    }
}

fn parse_header_block(
    buffer: &[u8],
    header: &mut HashMap<String, HeaderGroup>,
    current_pos: &mut u32,
    primary: &bool,
) -> Result<bool, String> {
    let mut buffer_idx: usize = 0;
    while buffer_idx < BLOCK_SIZE as usize {
        let line = &buffer[buffer_idx..buffer_idx + 80];
        buffer_idx += 80;
        match parse_header_line(line) {
            Ok((Some(kw), value, comment)) => {
                if let Some(e) = validate_header(&kw, &value, current_pos, header, primary) {
                    return Err(e);
                }
                if kw == "END" {
                    return Ok(true);
                }
                let header_entry = HeaderEntry {
                    value,
                    comment,
                    position: current_pos.clone(),
                };
                match header.get_mut(&kw) {
                    Some(HeaderGroup::Multi(v)) => {
                        v.push(header_entry);
                    }
                    None => {
                        header.insert(kw, HeaderGroup::Single(Some(header_entry)));
                    }
                    Some(HeaderGroup::Single(v)) => {
                        let old_entry = v.take().unwrap();
                        header.insert(kw, HeaderGroup::Multi(vec![old_entry, header_entry]));
                    }
                }
                *current_pos += 1;
            }
            Ok((None, _, _)) => {
                if let Some(e) =
                    validate_header("", &HeaderValue::Empty, current_pos, header, primary)
                {
                    return Err(e);
                }
            }
            Err(e) => {
                return Err(e);
            }
        }
    }

    Ok(false)
}

fn validate_header(
    kw: &str,
    value: &HeaderValue,
    current_pos: &u32,
    header: &HashMap<String, HeaderGroup>,
    primary: &bool,
) -> Option<String> {
    if *current_pos == 0 {
        let expected_kw = if *primary { "SIMPLE" } else { "XTENSION" };
        if kw != expected_kw {
            return Some(format!(
                "header starts with '{kw}' rather than '{expected_kw}'"
            ));
        }
        if let HeaderValue::Logical { v } = value {
            if !v {
                return Some(String::from("file does not conform to standard"));
            }
            None
        } else {
            return Some(String::from("unexpected value type for 'SIMPLE'"));
        }
    } else if *current_pos == 1 {
        if kw != "BITPIX" {
            return Some(format!(
                "2nd keyword in header is '{kw}' rather than 'BITPIX'"
            ));
        }
        if let HeaderValue::Integer { v } = value {
            if *v != 8 && *v != 16 && *v != 32 && *v != 64 && *v != -32 && *v != -64 {
                return Some(format!("unsupported value '{v}' for BITPIX"));
            }
            None
        } else {
            return Some(String::from("unexpected value for BITPIX"));
        }
    } else if *current_pos == 2 {
        if kw != "NAXIS" {
            return Some(format!(
                "3rd keyword in header is '{kw}' rather than 'NAXIS'"
            ));
        }
        if let HeaderValue::Integer { v } = value {
            if *v < 0 {
                return Some(String::from("NAXIS < 0"));
            }
            None
        } else {
            return Some(String::from("unexpected value for NAXIS"));
        }
    } else {
        if let Some(HeaderGroup::Single(Some(HeaderEntry {
            value: naxis_val, ..
        }))) = header.get("NAXIS")
        {
            if let HeaderValue::Integer { v: naxis } = naxis_val {
                if *current_pos as i64 > naxis + 2 {
                    return None;
                }
                let expected_naxis = format!("NAXIS{}", *current_pos - 2);
                if kw != expected_naxis {
                    return Some(format!(
                        "keyword #{} is '{kw}' rather than '{expected_naxis}'",
                        *current_pos
                    ));
                }
                None
            } else {
                panic!("unexpected type for NAXIS entry")
            }
        } else {
            panic!("NAXIS entry doesn't exist")
        }
    }
}

fn read_block(file: &mut impl Read, buf: &mut [u8]) -> Result<usize, String> {
    let capacity = (*buf).len();
    let mut remaining = capacity;
    let mut slice = buf;
    while remaining > 0 {
        match file.read(slice) {
            Ok(nread) if nread != 0 => {
                remaining -= nread;
                if remaining > 0 {
                    slice = slice[capacity - remaining..capacity].as_mut();
                }
            }
            Ok(_) => {
                break;
            }
            Err(e) => {
                return Err(format!("Error reading file: {e}"));
            }
        }
    }

    Ok(capacity - remaining)
}

fn parse_header_line(line: &[u8]) -> Result<(Option<String>, HeaderValue, Option<String>), String> {
    let str = String::from_utf8_lossy(line);
    let kw = decode_keyword(str.as_ref());
    let (value, comment) = match kw {
        Some(_) => decode_value(str.as_ref()),
        None => (HeaderValue::Empty, None),
    };
    Ok((kw, value, comment))
}

fn decode_keyword(line: &str) -> Option<String> {
    match line[0..8].find(' ') {
        Some(end) => {
            if end == 0 {
                None
            } else {
                Some(String::from(&line[0..end]))
            }
        }
        None => Some(String::from(&line[0..8])),
    }
}

fn decode_value(line: &str) -> (HeaderValue, Option<String>) {
    if !line[8..10].starts_with("= ") {
        return (HeaderValue::Empty, Some(String::from(&line[10..])));
    }
    let stripped = line[10..].trim_start();
    if stripped.starts_with('\'') {
        return decode_string(stripped);
    } else if stripped.starts_with('T') || stripped.starts_with('F') {
        return decode_bool(stripped);
    } else if stripped.starts_with('+')
        || stripped.starts_with('-')
        || stripped.chars().next().unwrap().is_ascii_digit()
    {
        return decode_number(stripped);
    } else if stripped.starts_with('(') {
        return decode_complex(stripped);
    }

    (HeaderValue::Empty, Some(String::from(&line[10..])))
}

fn decode_string(line: &str) -> (HeaderValue, Option<String>) {
    let mut value = String::with_capacity(line.len());
    let mut comment = String::with_capacity(line.len());
    let mut found_end = false;
    let mut in_comment = false;
    for ch in line[1..].chars() {
        if in_comment {
            comment.push(ch);
        } else if !found_end {
            if ch != '\'' {
                value.push(ch);
            } else {
                found_end = true;
            }
        } else if found_end && !in_comment && ch == '/' {
            in_comment = true;
        }
    }
    if !found_end {
        return (HeaderValue::Empty, Some(String::from(line)));
    }
    if in_comment {
        return (HeaderValue::Stringval { v: value }, Some(comment));
    }

    (HeaderValue::Stringval { v: value }, None)
}

fn decode_bool(line: &str) -> (HeaderValue, Option<String>) {
    let (slice, comment) = split_value_comment(line);
    if slice.chars().count() != 1 {
        return (HeaderValue::Empty, Some(String::from(line)));
    }
    let raw_value = slice.chars().next().unwrap();
    if raw_value == 'T' {
        return (HeaderValue::Logical { v: true }, comment);
    } else if raw_value == 'F' {
        return (HeaderValue::Logical { v: false }, comment);
    };

    (HeaderValue::Empty, Some(String::from(line)))
}

fn decode_number(line: &str) -> (HeaderValue, Option<String>) {
    let (slice, comment) = split_value_comment(line);
    if slice.contains('.') || slice.contains('e') || slice.contains('E') {
        match f64::from_str(slice) {
            Ok(value) => (HeaderValue::Float { v: value }, comment),
            _ => (HeaderValue::Empty, Some(String::from(line))),
        }
    } else {
        match i64::from_str(slice) {
            Ok(value) => (HeaderValue::Integer { v: value }, comment),
            _ => (HeaderValue::Empty, Some(String::from(line))),
        }
    }
}

fn decode_complex(line: &str) -> (HeaderValue, Option<String>) {
    let (slice, comment) = split_value_comment(line);
    if !slice.ends_with(')') {
        return (HeaderValue::Empty, Some(String::from(line)));
    }
    match slice.find(',') {
        Some(comma_idx) => {
            if let Ok(re) = f64::from_str(slice[1..comma_idx].trim()) {
                if let Ok(im) = f64::from_str(slice[comma_idx + 1..slice.len() - 1].trim()) {
                    (HeaderValue::Complex { re, im }, comment)
                } else {
                    (HeaderValue::Empty, Some(String::from(line)))
                }
            } else {
                (HeaderValue::Empty, Some(String::from(line)))
            }
        }
        None => (HeaderValue::Empty, Some(String::from(line))),
    }
}

fn split_value_comment(line: &str) -> (&str, Option<String>) {
    match line.find('/') {
        Some(comment_start) => (
            &line[0..comment_start].trim_end(),
            Some(String::from(&line[comment_start + 1..])),
        ),
        None => (line.trim_end(), None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_FILE_IMAGE: &str = "testdata/WFPC2-image-sample.fits";

    #[test]
    fn open_fits_file() {
        match open(TEST_FILE_IMAGE) {
            Ok(hdus) => {
                if let Some(primary_hdu) = hdus.get(0) {
                    let header = primary_hdu.header();
                    assert_header_bool(header, "SIMPLE", true);
                    assert_header_int(header, "BITPIX", -32);
                    assert_header_int(header, "NAXIS", 2);
                    assert_header_int(header, "NAXIS1", 100);
                    assert_header_int(header, "NAXIS2", 100);

                    if let Some(real_hdu) = (*primary_hdu).as_any().downcast_ref::<ImageHdu<f32>>()
                    {
                        if let Some(data) = &real_hdu.data {
                            assert_eq!(100 * 100, data.len())
                        } else {
                            panic!("No data in HDU")
                        }
                    } else {
                        panic!("expected HDU of f32");
                    }
                } else {
                    panic!("primary HDU is missing")
                }
            }
            Err(e) => {
                panic!("{e}");
            }
        }
    }

    fn assert_header_bool(header: &HashMap<String, HeaderGroup>, key: &str, expected: bool) {
        if let Some(HeaderGroup::Single(Some(HeaderEntry { value, .. }))) = header.get(key) {
            if let HeaderValue::Logical { v } = value {
                assert_eq!(expected, *v)
            } else {
                panic!("unexpected type for {key}")
            }
        } else {
            panic!("'{key}' entry missing")
        }
    }

    fn assert_header_int(header: &HashMap<String, HeaderGroup>, key: &str, expected: i64) {
        if let Some(HeaderGroup::Single(Some(HeaderEntry { value, .. }))) = header.get(key) {
            if let HeaderValue::Integer { v } = value {
                assert_eq!(expected, *v)
            } else {
                panic!("unexpected type for {key}")
            }
        } else {
            panic!("'{key}' entry missing")
        }
    }

    #[test]
    fn get_header_value() {
        let s = String::from("FITS");
        let sv = HeaderValue::Stringval { v: s.clone() };
        assert_eq!(s, sv.get::<String>().unwrap());

        let b = true;
        let bv = HeaderValue::Logical { v: b.clone() };
        assert_eq!(b, bv.get::<bool>().unwrap());

        let i: i64 = 42;
        let iv = HeaderValue::Integer { v: i.clone() };
        assert_eq!(i, iv.get::<i64>().unwrap());

        let f: f64 = 42.0;
        let fv = HeaderValue::Float { v: f.clone() };
        assert_eq!(f, fv.get::<f64>().unwrap())
    }
}
