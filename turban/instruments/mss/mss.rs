use byteorder::{LittleEndian, ReadBytesExt}; // 1.2.7
use std::cmp;
use std::f64::NAN;
use std::{
    fs::File,
    io::{self, *},
};

// pub fn convert_mrd(fname: &str, probe_conf_file: &str) -> HashMap<String, Vec<f64>> {
//     let raw = read_mrd(fname).unwrap();
//     let phys = convert_raw_mrd(raw, probe_conf_file);
//     phys
// }

#[derive(Debug, Clone, Copy)]
struct Record {
    identifier: u8,
    ch0: u16,
    ch1: u16,
    ch2: u16,
    ch3: u16,
    ch4: u16,
    ch5: u16,
    ch6: u16,
    ch7: u16,
}

impl Record {
    fn from_reader(rdr: &mut impl Read) -> io::Result<Self> {
        let identifier = rdr.read_u8()?;
        let ch0 = rdr.read_u16::<LittleEndian>()?;
        let ch1 = rdr.read_u16::<LittleEndian>()?;
        let ch2 = rdr.read_u16::<LittleEndian>()?;
        let ch3 = rdr.read_u16::<LittleEndian>()?;
        let ch4 = rdr.read_u16::<LittleEndian>()?;
        let ch5 = rdr.read_u16::<LittleEndian>()?;
        let ch6 = rdr.read_u16::<LittleEndian>()?;
        let ch7 = rdr.read_u16::<LittleEndian>()?;

        Ok(Record {
            identifier,
            ch0,
            ch1,
            ch2,
            ch3,
            ch4,
            ch5,
            ch6,
            ch7,
        })
    }
}

pub fn read_mrd_header(file: &mut File) -> Vec<u8> {
    // let mut reader = std::io::BufReader::new(file);

    // read header until stop byte
    let mut header_u8: Vec<u8> = Vec::new();
    for byte in file.bytes().map(|b| b.unwrap()) {
        header_u8.push(byte);
        if byte == 0x1A {
            break;
        }
    }
    header_u8
}

pub fn read_mrd(fname: &str) -> Result<Vec<[u16; 16]>> {
    let mut file = File::open(&fname).unwrap();
    // let mut reader = std::io::BufReader::new(file);

    read_mrd_header(&mut file);

    // seek start of date array
    for byte in (&mut file).bytes() {
        if byte.unwrap() == 0x01 {
            break;
        }
    }

    // read date
    let yy = file.read_u16::<LittleEndian>()?;
    let mm = file.read_u16::<LittleEndian>()?;
    let dd = file.read_u16::<LittleEndian>()?;
    let dow = file.read_u16::<LittleEndian>()?;
    let hh = file.read_u16::<LittleEndian>()?;
    let min = file.read_u16::<LittleEndian>()?;
    let ss = file.read_u16::<LittleEndian>()?;
    let ss100 = file.read_u16::<LittleEndian>()?;
    let date = vec![yy, mm, dd, dow, hh, min, ss, ss100];

    // determine length of data records
    let pos_cur = file.seek(SeekFrom::Current(0))?;
    let pos_eof = file.seek(SeekFrom::End(0))?;
    file.seek(SeekFrom::Start(pos_cur))?; // wind back
    let records_len = (pos_eof - pos_cur) / 17;

    // read data
    let dummy = Record {
        identifier: 0,
        ch0: 0,
        ch1: 0,
        ch2: 0,
        ch3: 0,
        ch4: 0,
        ch5: 0,
        ch6: 0,
        ch7: 0,
    };
    let mut records = vec![dummy; records_len as usize];

    for i in 0..records_len {
        // FIXME
        let record = Record::from_reader(&mut file)?;
        records[i as usize] = record;
        // let data = Record::from_reader(&mut reader)?;
    }

    // merge records into full data lines, depending on line identifier
    let records7: Vec<&Record> = records.iter().filter(|&&r| r.identifier == 7).collect();
    let records8: Vec<&Record> = records.iter().filter(|&&r| r.identifier == 8).collect();
    let data_len: usize = cmp::min(records7.len(), records8.len());
    let mut data = vec![[0; 16]; data_len];
    for (i, (&r7, &r8)) in records7.into_iter().zip(records8.into_iter()).enumerate() {
        let full = [
            r7.ch0, r7.ch1, r7.ch2, r7.ch3, r7.ch4, r7.ch5, r7.ch6, r7.ch7, r8.ch0, r8.ch1, r8.ch2,
            r8.ch3, r8.ch4, r8.ch5, r8.ch6, r8.ch7,
        ];
        data[i] = full;
    }

    Ok(data)
}

use serde_derive;
use serde_json::{self, from_reader, Value};
use std::collections::HashMap;

#[derive(Debug, serde_derive::Deserialize)]
struct Config {
    sensors: HashMap<String, Sensor>,
}
#[derive(Debug, serde_derive::Deserialize)]
struct Sensor {
    channel: usize,
    calibration_type: String,
    coefficients: [f64; 6],
}

fn polyval(x: f64, coefs: &[f64], pwrs: &[f64]) -> f64 {
    // evaluate a polynom on value x
    coefs
        .iter()
        .zip(pwrs)
        .map(|(c, pwr)| *c as f64 * x.powf(*pwr))
        .sum()
}

pub fn convert_raw_mrd(data: Vec<[u16; 16]>, probe_conf_file: &str) -> HashMap<String, Vec<f64>> {
    let file = File::open(probe_conf_file).unwrap();

    let cfg: Config = from_reader(file).expect("yo");

    let pwrs = vec![0., 1., 2., 3., 4., 5.];

    // make container for physical data
    let mut phys: HashMap<String, Vec<f64>> = HashMap::new();
    for (name, sensor) in cfg.sensors.iter() {
        let coefs = &sensor.coefficients;
        let channel_raw: Vec<u16> = data.iter().map(|v| v[sensor.channel]).collect();
        let channel_phys = match sensor.calibration_type.as_str() {
            "N" => channel_raw // shear probe sensitivity baked into coefficients
                .iter()
                .map(|v| polyval(*v as f64, coefs, &pwrs))
                .collect(),
            "P" => channel_raw
                .iter()
                .map(|v| polyval(*v as f64, &coefs[0..5], &pwrs[0..5]) + coefs[5])
                .collect(),
            "SHH" => channel_raw // NTC, NTCHP
                .iter()
                .map(|v| 1. / polyval((*v as f64).log(std::f64::consts::E), coefs, &pwrs) - 273.15)
                .collect(),
            "NFC" => channel_raw // chlorophyll fluorescence
                .iter()
                .map(|v| coefs[4] + coefs[5] * (coefs[0] + coefs[1] * (*v as f64))) // #TODO: can be simplified
                .collect(),
            _ => panic!("Unknown calibration type"), // should return Vec<NAN> instead
        };

        phys.insert(String::from(name), channel_phys);
    }
    phys
}
