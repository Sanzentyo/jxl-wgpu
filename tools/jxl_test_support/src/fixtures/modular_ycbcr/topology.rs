use std::str::SplitWhitespace;

#[derive(Debug)]
pub struct NativeTopology {
    pub meta_channels: usize,
    pub transform_count: usize,
    pub squeeze_channels: usize,
    /// Width, height, horizontal shift, vertical shift, including empty residuals.
    pub channels: Vec<[i64; 4]>,
}

impl NativeTopology {
    pub fn parse(text: &str) -> Self {
        let mut words = text.split_whitespace();
        let topology = Self::read(&mut words);
        assert!(words.next().is_none());
        topology
    }

    fn read(words: &mut SplitWhitespace<'_>) -> Self {
        let meta_channels = count(words);
        let channel_count = count(words);
        let transform_count = count(words);
        let squeeze_channels = count(words);
        assert!(meta_channels <= channel_count);
        Self {
            meta_channels,
            transform_count,
            squeeze_channels,
            channels: channels(words, channel_count),
        }
    }
}

#[derive(Debug)]
pub struct NativeSubstream {
    pub stream_index: u32,
    pub source: Vec<[i64; 4]>,
    pub transformed: NativeTopology,
}

impl NativeSubstream {
    pub fn parse_all(text: &str) -> Vec<Self> {
        let mut words = text.split_whitespace();
        let mut streams = Vec::new();
        while let Some(tag) = words.next() {
            assert_eq!(tag, "stream");
            let stream_index = words.next().unwrap().parse().unwrap();
            assert_eq!(words.next(), Some("source"));
            let channel_count = count(&mut words);
            let source = channels(&mut words, channel_count);
            assert_eq!(words.next(), Some("transformed"));
            let transformed = NativeTopology::read(&mut words);
            streams.push(Self {
                stream_index,
                source,
                transformed,
            });
        }
        assert!(!streams.is_empty());
        streams
    }
}

fn count(words: &mut SplitWhitespace<'_>) -> usize {
    words.next().unwrap().parse().unwrap()
}

fn channels(words: &mut SplitWhitespace<'_>, count: usize) -> Vec<[i64; 4]> {
    (0..count)
        .map(|_| std::array::from_fn(|_| words.next().unwrap().parse().unwrap()))
        .collect()
}
