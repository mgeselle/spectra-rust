use std::collections::HashMap;
use std::fs::{File, metadata};
use std::io::Read;

pub const BLOCK_SIZE: u64 = 2880;
pub enum HeaderValue {
    String {v: String},
    Logical {v: bool},
    Integer {v: i64},
    Float {v: f64},
    Complex {re: f64, im: f64},
}

pub struct ImageHdu<T> {
    header: HashMap<String, HeaderValue>,
    num_axes: u16,
    dimensions: Vec<u32>,
    data: Vec<T>,
}

pub enum Hdu {
    Int8 {hdu: ImageHdu<u8>},
    Int16 {hdu: ImageHdu<i16>},
    Int32 {hdu: ImageHdu<i32>},
    Int64 {hdu: ImageHdu<i64>},
    Float32 {hdu: ImageHdu<f32>},
    Float64 {hdu: ImageHdu<f64>}
    }

pub fn open(_file_name: &str) -> Result<Hdu, String> {

    let header : HashMap<String, HeaderValue> = HashMap::new();
    let _hdu : ImageHdu<u8> = ImageHdu {header, num_axes: 0, dimensions: vec![1], data: vec![] };
    Err(String::from("Not yet implemented"))
}

fn open_file(file_name: &str) -> Result<File, String> {
    let metares = metadata(file_name);
    match metares {
        Ok(metadata) => {
            if metadata.len() % BLOCK_SIZE != 0 {
                return Err(format!("File size isn't multiple of {BLOCK_SIZE}"))
            }

            let fileres = File::open(file_name);
            match fileres {
                Ok(file) => {
                    return Ok(file)
                }
                Err(e) => {
                    return Err(format!("Error opening file: {e}"))
                }
            }
        }
        Err(e) => {
            return Err(format!("Error getting metadata for file: {e}"))
        }
    }
}

fn read_block(file: &mut impl Read, buf: &mut [u8]) -> Result<usize, String> {
    let capacity = (*buf).len();
    let mut remaining = capacity;
    let mut slice = buf;
    while remaining > 0 {
        match file.read(slice) {
            Ok(nread) => {
                remaining -= nread;
                if remaining > 0 {
                    slice = slice[capacity - remaining..capacity].as_mut();
                }
            }
            Err(e) => {
                return Err(format!("Error reading file: {e}"))
            }
        }
    };

    Ok(capacity - remaining)
}

fn parse_header_line(line: &[u8]) -> Result<(Option<String>, Option<HeaderValue>, Option<String>), String> {
    let str = String::from_utf8_lossy(line);
    let kw = decode_keyword(str.as_ref());
    let value = match kw {
        Some(_) => {
            decode_value(str.as_ref())
        }
        None => {
            None
        }
    };
    let comment = decode_comment(str.as_ref());
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
        None => {
            Some(String::from(&line[0..8]))
        }
    }
}

fn decode_value(line: &str) -> Option<HeaderValue> {
    if !line[8..10].starts_with("= ") {
        return None
    }
    let stripped = line[10..].trim_start();
    if stripped.starts_with('\'') {
        return decode_string(line);
    } else if stripped.starts_with('T') || stripped.starts_with('F') {
        return decode_bool(stripped);
    } else if stripped.starts_with('+') || stripped.starts_with('-') || stripped.chars().next().unwrap().is_ascii_digit() {
        return decode_number(stripped)
    }

    None
}

fn decode_string(_line: &str) -> Option<HeaderValue> {
    None
}

fn decode_bool(_line: &str) -> Option<HeaderValue> {
    None
}

fn decode_number(_line: &str) -> Option<HeaderValue> {
    None
}

fn decode_comment(_line: &str) -> Option<String> {
    None
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_size_ok() {
        if let Err(msg ) = open_file("/home/mgeselle/astro/raw/latest/M_103/Light/R/M_103_Light_R_002.fits") {
            panic!("Error opening valid FITS file: {msg}")
        }
    }

    #[test]
    fn read_first_block() {
        let mut file = open_file("/home/mgeselle/astro/raw/latest/M_103/Light/R/M_103_Light_R_002.fits").unwrap();
        let mut buf: Vec<u8> = Vec::with_capacity(BLOCK_SIZE as usize);
        buf.resize(BLOCK_SIZE as usize, 0);
        match read_block(&mut file, buf.as_mut_slice()) {
            Ok(total_read) => {
                assert_eq!(buf.len(), total_read);
                if let Ok((kw, val, comment)) = parse_header_line(buf.as_slice()[0..79].as_ref()) {
                    assert_eq!(String::from("SIMPLE"), kw.unwrap())
                } else {
                    panic!("Error on parsing first header line")
                }
            }
            Err(e) => {
                panic!("{}", e)
            }
        };
    }

}