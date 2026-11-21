//! Frozen, versioned German compound morpheme lexicon.
//!
//! The lexicon is deliberately static: changing it changes retrieval results,
//! so [`GERMAN_LEXICON_VERSION`] is part of the persisted collection config and
//! a mismatch refuses to open an index.
//!
//! Every entry is lowercase and umlaut-folded (`ä -> ae`, `ö -> oe`,
//! `ü -> ue`, `ß -> ss`). Callers never have to normalize before consulting it:
//! [`is_lexicon_word`] folds its input first.
//!
//! The list is *permissive on purpose*. A false split costs ranking noise; a
//! missed split loses recall for an entire German compound family.

use std::collections::HashSet;
use std::sync::OnceLock;

/// Version of the frozen German morpheme lexicon.
///
/// Persisted in [`crate::v2::V2CollectionConfig::lexicon_version`]; a bump
/// requires a rebuild because indexed `body_parts` tokens change.
pub const GERMAN_LEXICON_VERSION: u32 = 1;

/// Shortest accepted compound part, in characters.
pub const MIN_PART_CHARS: usize = 4;

/// Fugenlaute (linking morphemes) that may sit between two compound parts.
const FUGENLAUTE: [&str; 5] = ["s", "es", "n", "en", "er"];

/// Curated, permissive list of German compound stems, nouns and roots.
///
/// Domain coverage: policy, funding, law, software, enterprise administration,
/// science, and everyday German. Entries are lowercase and umlaut-folded.
pub static GERMAN_LEXICON: &[&str] = &[
    // -- Policy, funding, law, administration -----------------------------
    "foerder",
    "foerderung",
    "foerderprogramm",
    "foerderrichtlinie",
    "programm",
    "programme",
    "richtlinie",
    "richtlinien",
    "verordnung",
    "verordnungen",
    "gesetz",
    "gesetze",
    "gesetzbuch",
    "verwaltung",
    "verwaltungs",
    "behoerde",
    "behoerden",
    "bundes",
    "ministerium",
    "ministerien",
    "ausschreibung",
    "ausschreibungen",
    "vergabe",
    "vergabeverfahren",
    "haushalt",
    "haushalts",
    "haushaltsplan",
    "budget",
    "etat",
    "zulage",
    "zulagen",
    "zuschlag",
    "zuschuss",
    "bewilligung",
    "genehmigung",
    "genehmigungsverfahren",
    "zuwendung",
    "subvention",
    "steuer",
    "steuern",
    "abgabe",
    "abgaben",
    "gebuehr",
    "gebuehren",
    "beitrag",
    "beitraege",
    "bescheid",
    "widerspruch",
    "klage",
    "urteil",
    "beschluss",
    "satzung",
    "ordnung",
    "vorschrift",
    "vorschriften",
    "regelung",
    "regelungen",
    "freigabe",
    "anordnung",
    "verfuegung",
    "ermaechtigung",
    "vollmacht",
    "vollzug",
    "aufsicht",
    "kontrolle",
    "pruefung",
    "pruefungs",
    "revision",
    "audit",
    "rechnung",
    "rechnungs",
    "rechnungswesen",
    "buchhaltung",
    "bilanz",
    "abschluss",
    "jahresabschluss",
    "quartal",
    "quartals",
    "halbjahr",
    "jahres",
    "geschaeftsjahr",
    // -- People, organisation, workplace ---------------------------------
    "mitarbeiter",
    "mitarbeiterin",
    "belegschaft",
    "personal",
    "personalabteilung",
    "abteilung",
    "abteilungs",
    "gruppe",
    "gruppen",
    "team",
    "teams",
    "stelle",
    "stellen",
    "stellenangebot",
    "angebot",
    "angebote",
    "bewerbung",
    "bewerbungen",
    "bewerber",
    "anschreiben",
    "lebenslauf",
    "erfahrung",
    "erfahrungen",
    "kompetenz",
    "kompetenzen",
    "qualifikation",
    "zeugnis",
    "zeugnisse",
    "arbeitsvertrag",
    "vertrag",
    "vertraege",
    "kuendigung",
    "einstellung",
    "entlassung",
    "arbeitszeit",
    "arbeitsplatz",
    "arbeits",
    "arbeit",
    "arbeitgeber",
    "arbeitnehmer",
    "betriebsrat",
    "gehalt",
    "gehaelter",
    "verguetung",
    "honorar",
    "lohn",
    "lohnabrechnung",
    "sozialversicherung",
    "versicherung",
    "krankenkasse",
    "rente",
    "pension",
    "urlaub",
    "fehlzeit",
    "krankmeldung",
    "fortbildung",
    "weiterbildung",
    "ausbildung",
    "bildung",
    "schulung",
    "seminar",
    "lehrgang",
    "studium",
    "hochschule",
    "universitaet",
    "institut",
    "fakultaet",
    "professor",
    "student",
    "studierende",
    "forschung",
    "wissenschaft",
    "wissenschaftlich",
    "labor",
    "experiment",
    "studie",
    "erhebung",
    "auswertung",
    // -- Software, data, infrastructure ----------------------------------
    "daten",
    "datenbank",
    "datenbanken",
    "datensatz",
    "datensaetze",
    "datenmodell",
    "datenschutz",
    "datensicherheit",
    "datenverarbeitung",
    "verarbeitung",
    "speicher",
    "speicherung",
    "speichern",
    "zwischenspeicher",
    "arbeitsspeicher",
    "massenspeicher",
    "system",
    "systeme",
    "systematik",
    "software",
    "softwareentwicklung",
    "hardware",
    "anwendung",
    "anwendungen",
    "anwendungsfall",
    "anwendungshandbuch",
    "handbuch",
    "handbuecher",
    "leitfaden",
    "dokumentation",
    "dokument",
    "dokumente",
    "unterlage",
    "unterlagen",
    "schnittstelle",
    "schnittstellen",
    "protokoll",
    "protokolle",
    "protokollierung",
    "netzwerk",
    "netzwerke",
    "netz",
    "server",
    "client",
    "dienst",
    "dienste",
    "dienstleistung",
    "cloud",
    "cluster",
    "container",
    "deployment",
    "version",
    "versionierung",
    "migration",
    "integration",
    "automatisierung",
    "automatisch",
    "konfiguration",
    "einstellung",
    "einstellungen",
    "parameter",
    "index",
    "indizierung",
    "suche",
    "suchmaschine",
    "volltext",
    "volltextsuche",
    "treffer",
    "ergebnis",
    "ergebnisse",
    "filter",
    "sortierung",
    "seite",
    "seiten",
    "archiv",
    "archivierung",
    "backup",
    "wiederherstellung",
    "sicherung",
    "sicherheit",
    "sicherheits",
    "verschluesselung",
    "schluessel",
    "passwort",
    "anmeldung",
    "abmeldung",
    "berechtigung",
    "berechtigungen",
    "rolle",
    "rollen",
    "zugang",
    "zugriff",
    "zugriffs",
    "rechte",
    "recht",
    "pflicht",
    "pflichten",
    "haftung",
    "risiko",
    "risiken",
    "schutz",
    "schutzbedarf",
    "schadsoftware",
    "schwachstelle",
    "angriff",
    "abwehr",
    "firewall",
    "fehler",
    "fehlerbehebung",
    "behebung",
    "stoerung",
    "ausfall",
    "wartung",
    "instandhaltung",
    "betrieb",
    "betriebs",
    "lizenz",
    "lizenzen",
    "kosten",
    "kostenstelle",
    "preis",
    "preise",
    "rabatt",
    "abrechnung",
    "performance",
    "leistung",
    "leistungen",
    "verfuegbarkeit",
    "zuverlaessigkeit",
    "skalierung",
    "support",
    "hotline",
    "ticket",
    "vorfall",
    "problem",
    "probleme",
    "loesung",
    "loesungen",
    "antwort",
    "antworten",
    "frage",
    "fragen",
    "thema",
    "themen",
    "bereich",
    "bereiche",
    "ebene",
    "ebenen",
    "stufe",
    "stufen",
    "phase",
    "phasen",
    "schritt",
    "schritte",
    "ablauf",
    "prozess",
    "prozesse",
    "workflow",
    "aufgabe",
    "aufgaben",
    "anforderung",
    "anforderungen",
    "bedingung",
    "bedingungen",
    "vorgabe",
    "vorgaben",
    "kriterium",
    "kriterien",
    "merkmal",
    "merkmale",
    "eigenschaft",
    "eigenschaften",
    "funktion",
    "funktionen",
    "modul",
    "module",
    "komponente",
    "komponenten",
    "werkzeug",
    "werkzeuge",
    "bibliothek",
    "paket",
    "pakete",
    "quelle",
    "quellen",
    "ziel",
    "ziele",
    "zielsetzung",
    "zweck",
    "nutzen",
    "mehrwert",
    "vorteil",
    "nachteil",
    // -- Strategy, analysis, projects ------------------------------------
    "strategie",
    "strategisch",
    "analyse",
    "analysen",
    "bewertung",
    "beurteilung",
    "empfehlung",
    "empfehlungen",
    "massnahme",
    "massnahmen",
    "aktions",
    "aktion",
    "aktionen",
    "plan",
    "plaene",
    "planung",
    "planungs",
    "projekt",
    "projekte",
    "projektplan",
    "projektleitung",
    "management",
    "leitung",
    "steuerung",
    "koordination",
    "kooperation",
    "beratung",
    "berater",
    "umsetzung",
    "einfuehrung",
    "einfuehrungs",
    "optimierung",
    "verbesserung",
    "weiterentwicklung",
    "entwicklung",
    "entwurf",
    "konzept",
    "konzepte",
    "modell",
    "modelle",
    "architektur",
    "struktur",
    "strukturen",
    "topologie",
    "methode",
    "methoden",
    "verfahren",
    "verfahrens",
    "vorgehen",
    "ansatz",
    "loesungsansatz",
    "messung",
    "metrik",
    "kennzahl",
    "kennzahlen",
    "statistik",
    "bericht",
    "berichte",
    "berichtswesen",
    "auskunft",
    "uebersicht",
    "ueberblick",
    "ueberwachung",
    "monitoring",
    "alarm",
    "warnung",
    "meldung",
    "benachrichtigung",
    "kommunikation",
    "besprechung",
    "notiz",
    "notizen",
    "vermerk",
    "hinweis",
    "hinweise",
    "anleitung",
    "leitlinie",
    // -- Business, market, finance ---------------------------------------
    "unternehmen",
    "unternehmens",
    "firma",
    "firmen",
    "konzern",
    "kunde",
    "kunden",
    "kundschaft",
    "auftrag",
    "auftraege",
    "auftraggeber",
    "auftragnehmer",
    "nachfrage",
    "markt",
    "marketing",
    "vertrieb",
    "umsatz",
    "gewinn",
    "verlust",
    "investition",
    "investor",
    "kapital",
    "finanz",
    "finanzen",
    "finanzierung",
    "zahlung",
    "zahlungen",
    "mahnung",
    "inkasso",
    "liquiditaet",
    "wirtschaft",
    "wirtschaftlich",
    "wirtschafts",
    "branche",
    "wettbewerb",
    "produkt",
    "produkte",
    "produktion",
    "fertigung",
    "qualitaet",
    "qualitaets",
    "qualitaetsmanagement",
    "standard",
    "norm",
    "normen",
    "zertifikat",
    "zertifizierung",
    "lieferung",
    "lieferant",
    "lager",
    "logistik",
    "transport",
    "versand",
    "bestellung",
    "einkauf",
    "verkauf",
    "handel",
    "handels",
    "kalkulation",
    "skonto",
    "bonus",
    // -- Governments, society, education ---------------------------------
    "regierung",
    "regierungs",
    "buerger",
    "buergerlich",
    "gesellschaft",
    "gesellschaftlich",
    "gemeinde",
    "kommune",
    "kommunal",
    "landkreis",
    "kreistag",
    "parlament",
    "ausschuss",
    "fraktion",
    "opposition",
    "wahl",
    "wahlen",
    "demokratie",
    "politik",
    "politisch",
    "staat",
    "staatlich",
    "republik",
    "verfassung",
    "grundgesetz",
    "reform",
    "initiative",
    "stiftung",
    "verein",
    "verband",
    "gewerkschaft",
    "kirche",
    "schule",
    "unterricht",
    "lehrer",
    "kindergarten",
    "erziehung",
    "jugend",
    "familie",
    "gesundheit",
    "gesundheits",
    "krankenhaus",
    "klinik",
    "arzt",
    "aerztlich",
    "pflege",
    "therapie",
    "medikament",
    "umwelt",
    "umweltschutz",
    "energie",
    "energieeffizienz",
    "klima",
    "klimaschutz",
    "verkehr",
    "mobilitaet",
    "wohnung",
    "wohnungsbau",
    "immobilie",
    "miete",
    "stadt",
    "staedtisch",
    "dorf",
    "region",
    "regional",
    "infrastruktur",
    "wasser",
    "abwasser",
    "strom",
    "versorgung",
    "entsorgung",
    // -- Science, technology, AI -----------------------------------------
    "technologie",
    "technik",
    "technisch",
    "information",
    "informations",
    "informatik",
    "intelligenz",
    "kuenstlich",
    "maschine",
    "maschinen",
    "lernen",
    "lern",
    "sprache",
    "sprachen",
    "sprach",
    "text",
    "texte",
    "semantik",
    "semantisch",
    "bedeutung",
    "vektor",
    "vektoren",
    "vektorraum",
    "aehnlichkeit",
    "einbettung",
    "training",
    "inferenz",
    "algorithmus",
    "algorithmen",
    "berechnung",
    "rechner",
    "rechen",
    "prozessor",
    "graphik",
    "visualisierung",
    "darstellung",
    "signal",
    "sensor",
    "messgeraet",
    "physik",
    "chemie",
    "biologie",
    "mathematik",
    "geometrie",
    "theorie",
    "hypothese",
    "beweis",
    "publikation",
    "zeitschrift",
    "konferenz",
    "vortrag",
    "referat",
    "zitat",
    "literatur",
    // -- Everyday nouns and modifiers ------------------------------------
    "anfang",
    "ende",
    "start",
    "beginn",
    "dauer",
    "frist",
    "termin",
    "kalender",
    "datum",
    "zeit",
    "zeitraum",
    "jahr",
    "monat",
    "woche",
    "tag",
    "stunde",
    "minute",
    "sekunde",
    "haeufig",
    "selten",
    "wichtig",
    "wichtigkeit",
    "notwendig",
    "moeglich",
    "moeglichkeit",
    "unmoeglich",
    "einfach",
    "schwierig",
    "komplex",
    "beispiel",
    "beispiele",
    "muster",
    "vorlage",
    "formular",
    "antrag",
    "antraege",
    "anschrift",
    "adresse",
    "kontakt",
    "telefon",
    "telefonnummer",
    "email",
    "brief",
    "briefkopf",
    "ansprechpartner",
    "name",
    "namen",
    "vorname",
    "nachname",
    "titel",
    "position",
    "verantwortung",
    "befugnis",
    "unterschrift",
    "uhrzeit",
    "ort",
    "gebaeude",
    "raum",
    "raeume",
    "etage",
    "flur",
    "tuer",
    "fenster",
    "auto",
    "fahrzeug",
    "fahrzeuge",
    "reise",
    "reisekosten",
    "flug",
    "hotel",
    "essen",
    "verpflegung",
    "kleidung",
    "material",
    "werkstoff",
    "geraet",
    "geraete",
    "anlage",
    "anlagen",
    "kasse",
    "kassen",
    "beleg",
    "belege",
    "quittung",
    "konto",
    "konten",
    "kontonummer",
    "bank",
    "bankverbindung",
    "karte",
    "karten",
    "ausweis",
    "pass",
    "fuehrerschein",
    "police",
    "schaden",
    "schadensfall",
    "unfall",
    "notfall",
    "hilfe",
    "rettung",
    "feuerwehr",
    "polizei",
];

fn lexicon() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| GERMAN_LEXICON.iter().copied().collect())
}

/// True when `word` is a known morpheme (case-insensitive, umlaut-folded).
pub fn is_lexicon_word(word: &str) -> bool {
    let folded = fold_umlauts(word);
    lexicon().contains(folded.as_str())
}

/// Lowercase `text` and expand German umlauts/eszett to ASCII digraphs.
///
/// This is the canonical form used by the lexicon and by every compound
/// lookup.
pub fn fold_umlauts(text: &str) -> String {
    fold_with_map(text).0
}

/// Fold `text`, also returning `map[i] = original char index of folded char i`
/// (with a trailing entry for the end of the word).
///
/// The map lets [`decompose_german_compound`] report both the folded and the
/// originally-spelled form of a part.
fn fold_with_map(text: &str) -> (String, Vec<usize>) {
    let mut folded = String::with_capacity(text.len() + 8);
    let mut map: Vec<usize> = Vec::with_capacity(text.len() + 1);
    for (original_index, original) in text.chars().enumerate() {
        for lower in original.to_lowercase() {
            let expansion = match lower {
                'ä' => "ae",
                'ö' => "oe",
                'ü' => "ue",
                'ß' => "ss",
                _ => {
                    folded.push(lower);
                    map.push(original_index);
                    continue;
                }
            };
            for character in expansion.chars() {
                folded.push(character);
                map.push(original_index);
            }
        }
    }
    map.push(text.chars().count());
    (folded, map)
}

/// Deterministically split a German compound into its morpheme parts.
///
/// The splitter minimizes the number of parts (dynamic programming over
/// character positions), requires every part to be at least
/// [`MIN_PART_CHARS`] characters long, and accepts the Fugenlaute `s`, `es`,
/// `n`, `en`, `er` between parts. Among equally small splits the longest part
/// wins, so the result never depends on iteration luck.
///
/// Returns both the umlaut-folded and the originally-spelled form of every
/// part, deduplicated, and never the whole word itself. A word without a valid
/// split yields an empty vector.
///
/// Documents are decomposed once at index time into the `body_parts` subfield.
/// Query terms are decomposed through [`crate::v2::body_part_terms`] and are
/// only ever matched against `body_parts`, never against the original text
/// fields — the split never rewrites what the user searched for.
pub fn decompose_german_compound(word: &str) -> Vec<String> {
    let (folded, map) = fold_with_map(word);
    let chars: Vec<char> = folded.chars().collect();
    let len = chars.len();
    if len < MIN_PART_CHARS * 2 {
        return Vec::new();
    }

    let mut best: Vec<Option<Vec<(usize, usize)>>> = vec![None; len + 1];
    best[0] = Some(Vec::new());

    for start in 0..len {
        let current = match best[start].clone() {
            Some(parts) => parts,
            None => continue,
        };
        let mut part_end = len;
        while part_end >= start + MIN_PART_CHARS {
            // The whole word is never accepted as a single part: a lexicon
            // entry that happens to be a compound must still expose its parts.
            let whole_word = start == 0 && part_end == len;
            if !whole_word && is_folded_word(&chars[start..part_end]) {
                relax(&mut best, part_end, &current, start, part_end);
                for connector in FUGENLAUTE {
                    let connector: Vec<char> = connector.chars().collect();
                    let next = part_end + connector.len();
                    // A trailing linking morpheme is never a valid ending.
                    if next < len && chars[part_end..next] == connector[..] {
                        relax(&mut best, next, &current, start, part_end);
                    }
                }
            }
            part_end -= 1;
        }
    }

    let parts = match best[len].take() {
        Some(parts) => parts,
        None => return Vec::new(),
    };
    if parts.len() < 2 {
        return Vec::new();
    }

    let original_chars: Vec<char> = word.chars().collect();
    let mut out: Vec<String> = Vec::with_capacity(parts.len() * 2);
    for (start, end) in parts {
        let folded_part: String = chars[start..end].iter().collect();
        if folded_part == folded {
            continue;
        }
        if !out.contains(&folded_part) {
            out.push(folded_part.clone());
        }
        let original_from = map[start];
        let original_to = map[end];
        let original_part: String = original_chars[original_from..original_to]
            .iter()
            .collect::<String>()
            .to_lowercase();
        if original_part != folded_part && !out.contains(&original_part) {
            out.push(original_part);
        }
    }
    out
}

fn is_folded_word(chars: &[char]) -> bool {
    let candidate: String = chars.iter().collect();
    lexicon().contains(candidate.as_str())
}

fn relax(
    best: &mut [Option<Vec<(usize, usize)>>],
    end: usize,
    current: &[(usize, usize)],
    start: usize,
    part_end: usize,
) {
    let mut candidate = current.to_vec();
    candidate.push((start, part_end));
    let replace = match &best[end] {
        None => true,
        Some(existing) => candidate.len() < existing.len(),
    };
    if replace {
        best[end] = Some(candidate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexicon_is_umlaut_folded() {
        for entry in GERMAN_LEXICON {
            assert_eq!(*entry, fold_umlauts(entry), "entry not folded: {entry}");
        }
        assert!(GERMAN_LEXICON.len() >= 500, "lexicon is too small");
    }

    #[test]
    fn decomposes_known_compounds() {
        let foerderprogramm = decompose_german_compound("Förderprogramm");
        assert!(foerderprogramm.contains(&"foerder".to_string()));
        assert!(foerderprogramm.contains(&"programm".to_string()));
        assert!(!foerderprogramm.contains(&"förderprogramm".to_lowercase()));

        let richtlinie = decompose_german_compound("Datenschutzrichtlinie");
        assert_eq!(
            richtlinie,
            vec!["datenschutz".to_string(), "richtlinie".to_string()]
        );

        assert_eq!(
            decompose_german_compound("Datenschutz"),
            vec!["daten".to_string(), "schutz".to_string()]
        );

        let zulage = decompose_german_compound("Forschungszulage");
        assert_eq!(zulage, vec!["forschung".to_string(), "zulage".to_string()]);

        let handbuch = decompose_german_compound("Anwendungshandbuch");
        assert_eq!(
            handbuch,
            vec!["anwendung".to_string(), "handbuch".to_string()]
        );
    }

    #[test]
    fn does_not_split_simple_words() {
        assert!(decompose_german_compound("Daten").is_empty());
        assert!(decompose_german_compound("Haus").is_empty());
    }

    #[test]
    fn folding_expands_german_characters() {
        assert_eq!(fold_umlauts("Größe"), "groesse");
        assert_eq!(fold_umlauts("FÖRDERUNG"), "foerderung");
    }

    #[test]
    fn returns_original_spelling_alongside_folded_parts() {
        let parts = decompose_german_compound("Förderprogramm");
        assert!(parts.contains(&"förder".to_string()), "parts: {parts:?}");
    }
}
