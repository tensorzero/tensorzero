local cjson = require "cjson"

-- Read the request body
ngx.req.read_body()
local body = ngx.req.get_body_data()

if not body then
    -- If body is not available, try to read from file
    local body_file = ngx.req.get_body_file()
    if body_file then
        local file = io.open(body_file, "r")
        if file then
            body = file:read("*all")
            file:close()
        end
    end
end

-- Only process if we have a body and it's a POST request with JSON content
if body and ngx.var.request_method == "POST" then
    local content_type = ngx.var.content_type or ""

    if string.find(content_type, "application/json") then
        -- Parse JSON
        local success, json_data = pcall(cjson.decode, body)

        if success and type(json_data) == "table" then
            -- Get USER from environment variable (optional)
            local user = os.getenv("USER") or ngx.var.user_env

            -- Only add the tensorzero::tags field if USER is set
            if user and user ~= "" then
                json_data["tensorzero::tags"] = {
                    user = user
                }
            end

            -- Convert back to JSON
            local new_body = cjson.encode(json_data)

            -- Set the new body
            ngx.req.set_body_data(new_body)

            -- Update content length
            ngx.req.set_header("Content-Length", string.len(new_body))
        end
    end
end
