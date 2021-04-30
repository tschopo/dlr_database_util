-- https://github.com/openstreetmap/osm2pgsql/tree/master/flex-config

-- Use JSON encoder
local json = require('dkjson')

local srid = 4326

local osm_railways = osm2pgsql.define_way_table('osm_railways', {

    { column = 'type',     type = 'text', not_null = true }, -- rail, light_rail, tram, narrow_gauge
    { column = 'status',     type = 'text', not_null = true }, -- active, proposed, construction, disused, razed
    { column = 'electrified',      type = 'text' }, -- yes, no, unknown
    { column = 'electrification',      type = 'text' }, -- contact_line, rail, 4th_rail, ground-level_power_supply, none, unknown
    { column = 'maxspeed', type = 'int' }, -- 
    { column = 'maxspeed_forward', type = 'int' }, -- 
    { column = 'maxspeed_backward', type = 'int' }, -- 
    { column = 'bridge',      type = 'text' }, --
   	{ column = 'bridge_type',      type = 'text' }, --
    { column = 'tunnel',      type = 'text' }, --
    { column = 'tunnel_type',      type = 'text' }, --
    { column = 'embankment',      type = 'text' }, -- 
    { column = 'cutting',      type = 'text' }, -- 
    { column = 'ref',      type = 'text' }, -- 
    { column = 'gauge',      type = 'text' }, -- 
    { column = 'traffic_mode',      type = 'text' }, -- 
    { column = 'service',      type = 'text' }, -- 
    { column = 'usage',      type = 'text' }, --
    { column = 'voltage',      type = 'text' }, --
    { column = 'frequency',      type = 'text' }, --    
    { column = 'tags', type = 'jsonb' },
    { column = 'geom', type = 'linestring' , projection = srid}
})

local osm_stations = osm2pgsql.define_node_table('osm_stations', {
	{ column = 'type',      type = 'text' , not_null = true }, -- station, halt, tram_stop, halt_position
	{ column = 'name',      type = 'text' }, -- bahnhofs name
	{ column = 'status',     type = 'text', not_null = true }, -- active, proposed, construction, disused, razed
	{ column = 'ref',  type = 'text' }, -- Ril-100-Kürzel (railway:ref)
	{ column = 'category',  type = 'text' }, -- railway:station_category
	{ column = 'elevation',  type = 'int' }, -- höhe über meeresspeiegel wenn vorhangen (tag: ele)
	{ column = 'operator',  type = 'text' },
	{ column = 'network',  type = 'text' },
    { column = 'tags', type = 'jsonb' },
    { column = 'geom', type = 'point' , projection = srid }
})

--[[ TODO add later
local osm_rail_relations = osm2pgsql.define_relation_table('osm_rail_relations', {
    { column = 'tags', type = 'jsonb' },
    { column = 'geom', type = 'multilinestring'}
})

local osm_rail_polygons = osm2pgsql.define_area_table('osm_rail_polygons', {
    { column = 'tags', type = 'jsonb' },
    { column = 'geom', type = 'geometry'},
    { column = 'area', type = 'area' },
})

local osm_rail_boundaries = osm2pgsql.define_relation_table('osm_rail_boundaries', {
    { column = 'tags', type = 'jsonb' },
    { column = 'geom', type = 'multilinestring'},
})
--]]


local delete_keys = {
    -- "mapper" keys
    'attribution',
    'comment',
    'created_by',
    'fixme',
    'note',
    'note:*',
    'odbl',
    'odbl:note',
    'source',
    'source:*',
    'source_ref',

    -- "import" keys

    -- Corine Land Cover (CLC) (Europe)
    'CLC:*',
   
    -- ngbe (ES)
    -- See also note:es and source:file above
    'ngbe:*',

    -- Friuli Venezia Giulia (IT)
    'it:fvg:*',   

    -- WroclawGIS (PL)
    'WroclawGIS:*',
    -- Naptan (UK)
    'naptan:*',

    -- EUROSHA (Various countries)
    'project:eurosha_2012',

    -- UrbIS (Brussels, BE)
    'ref:UrbIS',

    -- RUIAN (CZ)
    'ref:ruian:addr',
    'ref:ruian',
    'building:ruian:type',
    -- DIBAVOD (CZ)
    'dibavod:id',
    -- UIR-ADR (CZ)
    'uir_adr:ADRESA_KOD',

    -- FANTOIR (FR)
    'ref:FR:FANTOIR',

    -- 3dshapes (NL)
    '3dshapes:ggmodelk',
    -- AND (NL)
    'AND_nosr_r',

    -- OPPDATERIN (NO)
    'OPPDATERIN',
    -- Various imports (PL)
    'addr:city:simc',
    'addr:street:sym_ul',
    'building:usage:pl',
    'building:use:pl',
    -- TERYT (PL)
    'teryt:simc',

    -- RABA (SK)
    'raba:id',
 
    -- Address import from Bundesamt für Eich- und Vermessungswesen (AT)
    'at_bev:addr_date',

    -- misc
    'import',
    'import_uuid',
    'OBJTYPE',
    'SK53_bulk:load',
    'mml:class'
}

local clean_tags = osm2pgsql.make_clean_tags_func(delete_keys)



-- Parse a maxspeed value like "30" or "55 mph" and return a number in km/h
function parse_speed(input)
    if not input then
        return nil
    end

    local maxspeed = tonumber(input)

    -- If maxspeed is just a number, it is in km/h, so just return it
    if maxspeed then
        return maxspeed
    end

    return nil
end

function get_type(status, object)
	-- status must be list of statuses

	for i,s in ipairs(status) do
		t = object.tags[s..':railway']
		if t and t ~= '' then return t end
	end

	return 'rail'
end

function get_station_type(status, object)
    -- status must be list of statuses

    for i,s in ipairs(status) do
        t = object.tags[s..':railway']
        if t and t ~= '' then return t end
    end

    return 'station'
end


function osm2pgsql.process_node(object)

	-- only process railways
	if not object.tags.railway then
        return
    end

    if clean_tags(object.tags) then
        return
    end

     -- get status and type
    local type = object.tags.railway -- halt,station,tram_stop
    local status = 'active'

    if type == 'proposed' or type == 'planned' then
    	status = 'proposed'

    	-- now get type
		type = get_station_type({'proposed', 'planned'}, object)
    	
    elseif type == 'construction' then
    	status = 'construction'

    	-- now get type
		type = get_station_type({'construction'}, object)

    elseif type == 'disused' then
    	status = 'disused'

    	-- now get type
		type = get_station_type({'disused'}, object)

    elseif type == 'abandoned' or type == 'razed' or type == 'demolished' or type == 'removed' then
    	status = 'razed'

    	-- now get type
		type = get_station_type({'abandoned', 'razed', 'demolished', 'removed'}, object)
	elseif type == 'station'  then
    elseif type == 'halt' then
    elseif type == 'tram_stop' then
    elseif type == 'stop' then
    else
    	return
    end

    if type ~= 'station' and type ~= 'halt' and type ~= 'tram_stop' and type ~= 'stop' then
        return
    end

    -- https://wiki.openstreetmap.org/wiki/DE:OpenRailwayMap/Tagging	

    -- railway=station personenbahnhof
    -- railway=yard Rangier/güterbahnhof
    -- railway=service_station Betriebsbahnhof
    -- railway=halt haltestellte
    -- railway=tram_stop

    -- public_transport=stop_position + train=yes / tram=yes /light_rail=yes
    -- railway=stop wird auch manchmal verwendet (für nicht personenverkehr)

    -- name
    -- railway:ref banhofskrüzel
    -- uic_ref
    -- uic_name
    -- railway:station_category
    -- operator
    -- network
    -- ele // die höhe des bahnhöfs meerespiegel
    -- start_date
    -- end_date


    -- disused:railway=station / abandoned

    ----------

    -- # Abzweigungen (Verbindungsstelle zweier Bahnstrecken)
    -- railway=junction
    -- name
    --railway:ref=*  Ril-100-Kürzel
    -- operator
    -- ele

    -- # Überleitstelle (Züge das Gleis innerhalb einer Strecke wechseln können)
    -- railway=crossover
    -- name
    -- railway:ref=* Ril-100-Kürzel
    -- operator=*
    -- ele=*

    -- # railway=spur_junction

    -- # tankstellen TODO kann auch fläche sein. postprocessing: alles in eine tabelle, fläche in punkt umwandeln
    -- railway=fuel
    -- area=yes
    -- building=yes

    -- # oberleitungsmasten
    -- power=catenary_mast 

    ref_tag = 'railway:ref'

    -- get ref with operator if null such as railway:ref:DBAG
    ref = object.tags[ref_tag]

    if ref == nil then
        for key, value in pairs(object.tags) do
            if key:sub(1, #ref_tag) == ref_tag then
                ref = object.tags[key]
            end
        end
    end


    osm_stations:add_row({
    	type = type,
    	name = object.tags.name,
    	status = status,
    	ref = ref,
    	category = object.tags['railway:station_category'],
    	elevation = object.tags.ele,
    	operator = object.tags.operator,
    	network = object.tags.network,
        tags = json.encode(object.tags), -- store tags in jsonb format
    })

end

-- lua cannot search lists, convert to table with key as value
local t = {'covered', 'viaduct', 'aqueduct', 'cantilever', 'boardwalk', 'movable', 'trestle', 'low_water_crossing', 'lift', 'abandoned', }
local bridge_types = {}
for _, k in ipairs(bridge_types) do
    bridge_types[k] = 1
end

function osm2pgsql.process_way(object)

	-- only process railways
	if not object.tags.railway then
        return
    end

    -- if no rags after clean
    if clean_tags(object.tags) then
        return
    end

    -- process the tags

    -- get status and type
    local type = object.tags.railway -- rail, light_rail, narrow_gauge, tram ; or status
    local status = 'active'

    if type == 'proposed' or type == 'planned' then
    	status = 'proposed'

    	-- now get type
		type = get_type({'proposed', 'planned'}, object)
    	
    elseif type == 'construction' then
    	status = 'construction'

    	-- now get type
		type = get_type({'construction'}, object)

    elseif type == 'disused' then
    	status = 'disused'

    	-- now get type
		type = get_type({'disused'}, object)

    elseif type == 'abandoned' or type == 'razed' or type == 'demolished' or type == 'removed' then
    	status = 'razed'

    	-- now get type
		type = get_type({'abandoned', 'razed', 'demolished', 'removed'}, object)

    elseif type == 'rail'  then
    elseif type == 'light_rail' then
    elseif type == 'narrow_gauge' then
    elseif type == 'tram' then
    else
    	return
    end

    -- get electrification
    local electrified = object.tags.electrified -- true / false
    local electrification = 'contact_line'

    if electrified == 'no' then
    	electrified = 'no' 
    	electrification = 'none'
    elseif electrified == 'yes' then
    	electrified = 'yes' 
    	electrification = 'contact_line'
    elseif electrified == 'contact_line' then
    	electrified = 'yes' 
    	electrification = 'contact_line'
    elseif electrified == 'rail' then
    	electrified = 'yes' 
    	electrification = 'rail'
    elseif electrified == '4th_rail' then
    	electrified = 'yes' 
    	electrification = '4th_rail'
    elseif electrified == 'ground-level_power_supply' then
    	electrified = 'yes' 
    	electrification = 'ground-level_power_supply'
    else
    	electrified =  'unknown'
    	electrification = 'unknown'
    end

    -- frequency and voltage
    local voltage = object.tags.voltage
    local frequency = object.tags.frequency
   
    -- get speed
    local maxspeed = parse_speed(object.tags.maxspeed)
    local maxspeed_forward =  parse_speed(object.tags['maxspeed:forward'])
    local maxspeed_backward =  parse_speed(object.tags['maxspeed:backward'])

    -- bridge, tunnel, embankment, cutting
    local bridge = object.tags.bridge
    local bridge_type = nil

    if bridge=='yes' then
    	bridge_type = nil
    elseif bridge == 'maybe' then
    	bridge = 'yes'
    	bridge_type = nil
    elseif bridge_types[bridge] then
    	bridge_type = bridge
    	bridge = 'yes'
    else 
    	bridge_type = nil
    	bridge = 'no'
    end

    local tunnel = object.tags.tunnel
    local tunnel_type = nil

    if tunnel=='yes' then
    	tunnel_type = nil
    elseif tunnel == 'maybe' then
    	tunnel = 'yes'
    	tunnel_type = nil
    elseif tunnel == 'covered' or tunnel == 'passage' or tunnel == 'building_passage' or tunnel == 'culvert' or tunnel == 'abandoned' or tunnel == 'razed' then
    	tunnel_type = tunnel
    	tunnel = 'yes'
    else 
    	tunnel_type = nil
    	tunnel = 'no'
    end


    local embankment = object.tags.embankment
    -- {2,both,both_sides,left,levee,no,right,two_sided,yes,NULL}
    if embankment == "" then
    	embankment = 'no'
    elseif not embankment then
    	embankment = 'no'
    elseif embankment == "no" then
    	embankment = 'no'
    else
    	embankment = 'yes'
    end

    local cutting = object.tags.cutting
    -- {both,left,no,razed,right,yes,yy,NULL}
     if cutting == "" then
    	cutting = 'no'
    elseif not cutting then
    	cutting = 'no'
    elseif cutting == "no" then
    	cutting = 'no'
    else
    	cutting = 'yes'
    end

    -- streckennummer
    local ref = object.tags.ref

    local gauge = object.tags.gauge
    local traffic_mode = object.tags["traffic_mode"]

    -- service
    local service = object.tags.service
    local usage = object.tags.usage

    -- operator

    -- railway:preferred_direction

    -- highspeed = yes

    -- railway:tilting=yes

    -- 

    osm_railways:add_row({
    	type = type,
    	status = status,
    	electrified =  electrified,
    	electrification = electrification,
    	maxspeed = maxspeed,
    	maxspeed_forward = maxspeed_forward,
    	maxspeed_backward = maxspeed_backward,
    	bridge = bridge,
    	bridge_type = bridge_type,
    	tunnel = tunnel,
    	tunnel_type = tunnel_type,
    	embankment = embankment,
    	cutting = cutting,
    	ref = ref,
    	gauge = gauge,
    	traffic_mode = traffic_mode,
    	service = service,
    	usage = usage,
    	voltage = voltage,
    	frequency = frequency,
        tags = json.encode(object.tags), -- store tags in jsonb format
        geom = { create = 'line' }
    })
end

function osm2pgsql.process_relation(object)

	-- TODO add later
	if true then
		return
	end

    if clean_tags(object.tags) then
        return
    end

    local type = object.tags.type

    if type == 'route' then

    	-- # VzG Strecken
    	-- route=tracks
    	-- ref vzg streckennummer
    	-- operator 
    	-- from
    	-- to
    	-- via
    	-- name
    	-- role: ! widersprüchliche angaben orm sagt nur gleise in relation, keine stops


    	-- # Kursbuch Strecken
    	-- route=railway
    	-- ref KBS nummer
    	-- from 
    	-- to
    	-- via
    	-- name

    	-- bahnhöfe: role stop ! widersprüchliche angaben orm sagt nur gleise in relation, keine stops
    		-- railway=station bzw. railway=halt

    	-- # Linien 
    	-- route=train/tram/light_rail
    	-- from
    	-- to
    	-- via
    	-- name <Linientyp/Zugattung> <Liniennummer>: <Start> => <Ziel>
    	-- ref (z.B RE7)
    	-- ref:<Bezeichner> // Liniennummer, die vom Betreiber oder Verkehrsverbund vergeben wird, z.B. ref:VRN=R 85
    	-- operator
    	-- network
    	-- interval
    	-- duration
    	-- colour
    	-- service // Zuggattung: tourism/night/car_shuttle/car/commuter/regional/long_distance/high_speed
    	-- public_transport:version

    	-- roles: stop public_transport=stop_position

        osm_rail_relations:add_row({
       		tags = json.encode(object.tags), -- store tags in jsonb format
            geom = { create = 'line' }
        })
        return
    end

    if type == 'boundary' or (type == 'multipolygon' and object.tags.boundary) then
        osm_rail_boundaries:add_row({
       		tags = json.encode(object.tags), -- store tags in jsonb format
            geom = { create = 'line' }
        })
        return
    end

    if object.tags.type == 'multipolygon' then
        osm_rail_polygons:add_row({
       		tags = json.encode(object.tags), -- store tags in jsonb format
            geom = { create = 'area' }
        })
    end
end
