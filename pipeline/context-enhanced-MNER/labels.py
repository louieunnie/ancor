# import torch
gmner_label_list = [
   "O",
   "B-PER", "I-PER",
   "B-ORG", "I-ORG",
   "B-LOC", "I-LOC",
   "B-OTHER", "I-OTHER"
]
mner_label_list = [
   "O",
   "B-PER", "I-PER",
   "B-ORG", "I-ORG",
   "B-LOC", "I-LOC",
   "B-MISC", "I-MISC"
]
coarse_label_list = [
    "O",
    "B-person", "I-person",
    "B-organization", "I-organization",
    "B-location", "I-location",
    "B-event", "I-event",
    "B-product", "I-product",
    "B-art", "I-art",
    "B-other", "I-other",
    "B-building", "I-building"
]
fine_label_list = [
    "B-actor", "B-animal", "B-art_other", "B-artist", "B-athlete", "B-author", "B-award", "B-band", "B-brand_name_products",
    "B-building_other", "B-businessman", "B-character", "B-city", "B-coach", "B-company", "B-continent", "B-country", 
    "B-cultural_place", "B-director", "B-educational_institution", "B-entertainment_place", "B-event_other", "B-festival", 
    "B-film_and_television_works", "B-game", "B-government_agency", "B-intellectual", "B-journalist", "B-location_other", 
    "B-magazine", "B-medical_thing", "B-music", "B-musician", "B-news_agency", "B-ordinance", "B-organization_other", 
    "B-park", "B-person_other", "B-political_party", "B-politician", "B-product_other", "B-road", "B-social_organization", 
    "B-software", "B-sports_event", "B-sports_facility", "B-sports_league", "B-sports_team", "B-state", "B-website", 
    "B-written_work",
    "I-actor", "I-animal", "I-art_other", "I-artist", "I-athlete", "I-author", "I-award", "I-band", "I-brand_name_products",
    "I-building_other", "I-businessman", "I-character", "I-city", "I-coach", "I-company", "I-continent", "I-country", 
    "I-cultural_place", "I-director", "I-educational_institution", "I-entertainment_place", "I-event_other", "I-festival", 
    "I-film_and_television_works", "I-game", "I-government_agency", "I-intellectual", "I-journalist", "I-location_other", 
    "I-magazine", "I-medical_thing", "I-music", "I-musician", "I-news_agency", "I-ordinance", "I-organization_other", 
    "I-park", "I-person_other", "I-political_party", "I-politician", "I-product_other", "I-road", "I-social_organization", 
    "I-software", "I-sports_event", "I-sports_facility", "I-sports_league", "I-sports_team", "I-state", "I-website", 
    "I-written_work",
    "O"
]
coarse_fine_tree = {
    'location': ['city','country','state','continent','location_other','park','road'],
    'building': ['building_other','cultural_place','entertainment_place','sports_facility'],
    'organization': ['company','educational_institution','band','government_agency','news_agency','organization_other','political_party','social_organization','sports_league','sports_team'],
    'person': ['politician','musician','actor','artist','athlete','author','businessman','character','coach','director','intellectual','journalist','person_other'],
    'other': ['animal','award','medical_thing','website','ordinance'],
    'art': ['art_other','film_and_television_works','magazine','music','written_work'],
    'event': ['event_other','festival','sports_event'],
    'product': ['brand_name_products','game','product_other','software']
}
super_coarse_tree = {
    "LOC" : ["location", "building"],
    "ORG" : ["organization"],
    "PER" : ["person"],
    "OTHER" : ["other", "art", "event", "product"]
}
def build_label_mappings(labels):
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    types = sorted({t.split("-", 1)[1] for t in labels if t != "O"})
    label2typeid = {t: i for i, t in enumerate(types)}
    typeid2label = {i: t for t, i in label2typeid.items()}
    
    return label2id, id2label, label2typeid, typeid2label

def build_maps(coarse_label2id, fine_label2id, tree):
    c2f, f2c = {}, {}
    for c_lbl, f_lbls in tree.items():
        c_id = coarse_label2id[c_lbl]
        c2f[c_id] = [fine_label2id[f] for f in f_lbls]
        for f in f_lbls:
            f2c[fine_label2id[f]] = c_id
    return c2f, f2c

