#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <random>
#include <sstream>
#include <fstream>
#include <math.h>
#include <omp.h>

#include "Camera.h"
#include "FileSystemUtils.h"

// Asset Importer
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// Intel Embree library
#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>
#include <embree4/rtcore_common.h>

#define M_PI 3.14159265358979323846264338327950288

void APIENTRY MessageCallback(GLenum source,
    GLenum type,
    GLuint id,
    GLenum severity,
    GLsizei length,
    const GLchar* message,
    const void* userParam)
{
    std::cerr << "GL CALLBACK: " << (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "")
        << " type = " << type
        << ", severity = " << severity
        << ", message = " << message << std::endl;
}

// Constants and global variables
const int WIDTH = 2560;
const int HEIGHT = 1080;
float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
bool firstMouse = true;
float deltaTime = 0.0f; // Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame
double previousTime = 0.0;
int frameCount = 0;

// Embree 
RTCDevice device = nullptr;
RTCScene embreeScene = nullptr;

Camera camera(glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), -180.0f, 0.0f, 6.0f, 0.1f, 45.0f, 0.1f, 500.0f);

const char* vertexShaderSource = R"(
    #version 430 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec2 aTexCoords;

    // SH coefficients
    layout (location = 5) in vec3 shCoeffs0;
    layout (location = 6) in vec3 shCoeffs1;
    layout (location = 7) in vec3 shCoeffs2;
    layout (location = 8) in vec3 shCoeffs3;
    layout (location = 9) in vec3 shCoeffs4;
    layout (location = 10) in vec3 shCoeffs5;
    layout (location = 11) in vec3 shCoeffs6;
    layout (location = 12) in vec3 shCoeffs7;
    layout (location = 13) in vec3 shCoeffs8;

    out vec2 TexCoords;
    out vec3 v_shCoeffs[9];

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        TexCoords = aTexCoords;
        gl_Position = projection * view * model * vec4(aPos, 1.0);

        // Pass SH coefficients to fragment shader
        v_shCoeffs[0] = shCoeffs0;
        v_shCoeffs[1] = shCoeffs1;
        v_shCoeffs[2] = shCoeffs2;
        v_shCoeffs[3] = shCoeffs3;
        v_shCoeffs[4] = shCoeffs4;
        v_shCoeffs[5] = shCoeffs5;
        v_shCoeffs[6] = shCoeffs6;
        v_shCoeffs[7] = shCoeffs7;
        v_shCoeffs[8] = shCoeffs8;
    }
)";

const char* fragmentShaderSource = R"(
    #version 430 core
    out vec4 FragColor;

    in vec2 TexCoords;
    in vec3 v_shCoeffs[9];

    uniform vec3 lightSHCoeffs[9];
    uniform sampler2D diffuseTexture;

    void main() {
        vec3 color = vec3(0.0);
        for (int i = 0; i < 9; ++i) {
            color += v_shCoeffs[i] * lightSHCoeffs[i];
        }
        vec3 albedo = texture(diffuseTexture, TexCoords).rgb;
        FragColor = vec4(color * albedo, 1.0);
    }
)";

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.processKeyboardInput(GLFW_KEY_W, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.processKeyboardInput(GLFW_KEY_S, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.processKeyboardInput(GLFW_KEY_A, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.processKeyboardInput(GLFW_KEY_D, deltaTime);
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    static bool firstMouse = true;
    static float lastX = WIDTH / 2.0f;
    static float lastY = HEIGHT / 2.0f;

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // Reversed since y-coordinates range from bottom to top
    lastX = xpos;
    lastY = ypos;

    camera.processMouseMovement(xoffset, yoffset);
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    camera.processMouseScroll(static_cast<float>(yoffset));
}

// Utility function to load textures using stb_image or similar
GLuint loadTextureFromFile(const char* path, const std::string& directory);

struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
    glm::vec3 Tangent;
    glm::vec3 Bitangent;
    glm::vec3 shCoefficients[9]; // For 3rd-order SH
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    mutable unsigned int VAO;  // Mark as mutable to allow modification in const functions
    GLuint diffuseTexture;  // Store diffuse texture ID
    GLuint normalMapTexture;

    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, GLuint diffuseTexture, GLuint normalMapTexture)
        : vertices(vertices), indices(indices), diffuseTexture(diffuseTexture), normalMapTexture(normalMapTexture) {
    }

    void setupMesh() const {
        // Set up the VAO, VBO, and EBO as before
        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);

        unsigned int VBO, EBO;
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        // Vertex Positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        // Vertex Normals
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
        // Vertex Texture Coords (Diffuse)
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
        // Tangents
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));
        // Bitangents
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Bitangent));

        // For SH coefficients
        for (int i = 0; i < 9; ++i) {
            glEnableVertexAttribArray(5 + i);
            glVertexAttribPointer(5 + i, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                (void*)(offsetof(Vertex, shCoefficients) + sizeof(glm::vec3) * i));
        }

        glBindVertexArray(0);
    }

    void Draw(GLuint shaderProgram) const {
        // Bind diffuse texture
        GLint diffuseLoc = glGetUniformLocation(shaderProgram, "diffuseTexture");
        if (diffuseLoc != -1) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, diffuseTexture);
            glUniform1i(diffuseLoc, 0);
        }

        // Bind normal map texture
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, normalMapTexture);
        glUniform1i(glGetUniformLocation(shaderProgram, "normalMap"), 1);

        // Bind VAO and draw the mesh
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
};

std::vector<Mesh> loadModel(const std::string& path) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path,
        aiProcess_Triangulate | aiProcess_FlipUVs |
        aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices |
        aiProcess_CalcTangentSpace | aiProcess_PreTransformVertices |
        aiProcess_MakeLeftHanded | aiProcess_FlipWindingOrder);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
        return {};
    }

    GLuint diffuseTexture = 0;
    GLuint normalMapTexture = 0;
    std::vector<Mesh> meshes;

    for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[i];
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;

        // Process vertices and indices
        for (unsigned int j = 0; j < mesh->mNumVertices; j++) {
            Vertex vertex;
            vertex.Position = glm::vec3(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);
            vertex.Normal = glm::vec3(mesh->mNormals[j].x, mesh->mNormals[j].y, mesh->mNormals[j].z);

            if (mesh->mTextureCoords[0]) {
                vertex.TexCoords = glm::vec2(mesh->mTextureCoords[0][j].x, mesh->mTextureCoords[0][j].y);
            }
            else {
                vertex.TexCoords = glm::vec2(0.0f, 0.0f);
            }

            // Get tangent and bitangent from Assimp
            if (mesh->HasTangentsAndBitangents()) {
                vertex.Tangent = glm::vec3(mesh->mTangents[j].x, mesh->mTangents[j].y, mesh->mTangents[j].z);
                vertex.Bitangent = glm::vec3(mesh->mBitangents[j].x, mesh->mBitangents[j].y, mesh->mBitangents[j].z);
            }
            else {
                vertex.Tangent = glm::vec3(0.0f);
                vertex.Bitangent = glm::vec3(0.0f);
            }

            vertices.push_back(vertex);
        }

        for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
            aiFace face = mesh->mFaces[j];
            for (unsigned int k = 0; k < face.mNumIndices; k++) {
                indices.push_back(face.mIndices[k]);
            }
        }

        // Load the material
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

        if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
            aiString str;
            material->GetTexture(aiTextureType_DIFFUSE, 0, &str);
            std::string texturePath = FileSystemUtils::getAssetFilePath(std::string(str.C_Str()));
            diffuseTexture = loadTextureFromFile(texturePath.c_str(), "");
        }

        if (material->GetTextureCount(aiTextureType_NORMALS) > 0) {
            // Existing code to load normal map from the model file
            aiString str;
            material->GetTexture(aiTextureType_NORMALS, 0, &str);
            std::string texturePath = FileSystemUtils::getAssetFilePath(std::string(str.C_Str()));
            normalMapTexture = loadTextureFromFile(texturePath.c_str(), "");
        }

        meshes.push_back(Mesh(vertices, indices, diffuseTexture, normalMapTexture));
    }

    return meshes;
}

GLuint loadTextureFromFile(const char* path, const std::string&) {
    GLuint textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    unsigned char* data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data) {
        GLenum format;
        if (nrComponents == 1)
            format = GL_RED;
        else if (nrComponents == 3)
            format = GL_RGB;
        else if (nrComponents == 4)
            format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else {
        std::cerr << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}

float randomFloat(std::default_random_engine& engine,
    std::uniform_real_distribution<float>& distribution) {
    return distribution(engine);
}

void computeSphericalHarmonicsBasisFunctions(const glm::vec3& dir, std::vector<float>& Ylm) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    Ylm.resize(9);

    // l = 0, m = 0
    Ylm[0] = 0.282095f;

    // l = 1
    Ylm[1] = 0.488603f * y;
    Ylm[2] = 0.488603f * z;
    Ylm[3] = 0.488603f * x;

    // l = 2
    Ylm[4] = 1.092548f * x * y;
    Ylm[5] = 1.092548f * y * z;
    Ylm[6] = 0.315392f * (3.0f * z * z - 1.0f);
    Ylm[7] = 1.092548f * x * z;
    Ylm[8] = 0.546274f * (x * x - y * y);
}

glm::vec3 sampleHemisphere(const glm::vec3& normal,
    std::default_random_engine& engine,
    std::uniform_real_distribution<float>& distribution) {
    float u1 = distribution(engine);
    float u2 = distribution(engine);

    float r = sqrt(u1);
    float theta = 2.0f * M_PI * u2;

    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(1.0f - u1);

    // Create a coordinate system (Tangent Space)
    glm::vec3 tangent, bitangent;
    if (fabs(normal.x) > fabs(normal.z))
        tangent = glm::vec3(-normal.y, normal.x, 0.0f);
    else
        tangent = glm::vec3(0.0f, -normal.z, normal.y);
    tangent = glm::normalize(tangent);
    bitangent = glm::cross(normal, tangent);

    // Transform sample to world space
    glm::vec3 sampleWorld = x * tangent + y * bitangent + z * normal;
    return glm::normalize(sampleWorld);
}

void computeLightSHCoefficients(const glm::vec3& lightDir, const glm::vec3& lightIntensity, std::vector<glm::vec3>& lightSHCoeffs) {
    const int numCoeffs = 9;
    lightSHCoeffs.resize(numCoeffs);

    std::vector<float> Ylm(numCoeffs);
    computeSphericalHarmonicsBasisFunctions(-lightDir, Ylm); // Negate for incoming direction

    for (int i = 0; i < numCoeffs; ++i) {
        lightSHCoeffs[i] = lightIntensity * Ylm[i];
    }
}

void buildAccelerationStructure(const Mesh& mesh) {
    // Create a new triangle geometry
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Set vertex buffer
    size_t numVertices = mesh.vertices.size();
    float* vertices = (float*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
        RTC_FORMAT_FLOAT3, sizeof(float) * 3, numVertices);

    // Copy vertex data
    for (size_t i = 0; i < numVertices; ++i) {
        vertices[3 * i + 0] = mesh.vertices[i].Position.x;
        vertices[3 * i + 1] = mesh.vertices[i].Position.y;
        vertices[3 * i + 2] = mesh.vertices[i].Position.z;
    }

    // Set index buffer
    size_t numTriangles = mesh.indices.size() / 3;
    unsigned int* indices = (unsigned int*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0,
        RTC_FORMAT_UINT3, sizeof(unsigned int) * 3, numTriangles);

    // Copy index data
    for (size_t i = 0; i < numTriangles; ++i) {
        indices[3 * i + 0] = mesh.indices[3 * i + 0];
        indices[3 * i + 1] = mesh.indices[3 * i + 1];
        indices[3 * i + 2] = mesh.indices[3 * i + 2];
    }

    // Commit geometry and attach to scene
    rtcCommitGeometry(geom);
    rtcAttachGeometry(embreeScene, geom);
    rtcReleaseGeometry(geom); // Geometry can be released after attaching
}

bool isOccluded(const glm::vec3& origin, const glm::vec3& dir) {
    // Initialize ray
    RTCRay ray;
    ray.org_x = origin.x;
    ray.org_y = origin.y;
    ray.org_z = origin.z;
    ray.dir_x = dir.x;
    ray.dir_y = dir.y;
    ray.dir_z = dir.z;
    ray.tnear = 0.001f; // Avoid self-intersection
    ray.tfar = std::numeric_limits<float>::infinity(); // Set tfar to a large value
    ray.time = 0.0f;
    ray.mask = -1;
    ray.id = 0;
    ray.flags = 0;

    // Initialize occlusion arguments
    RTCOccludedArguments args;
    rtcInitOccludedArguments(&args);

    // Create a thread-local context
    RTCRayQueryContext context;
    rtcInitRayQueryContext(&context);
    args.context = &context;

    // Perform occlusion query
    rtcOccluded1(embreeScene, &ray, &args);

    // Check for errors
    RTCError error = rtcGetDeviceError(device);
    if (error != RTC_ERROR_NONE) {
        std::cerr << "Embree error during occlusion query: " << rtcGetErrorString(error) << std::endl;
        return false; // Or handle the error as appropriate
    }

    // If the ray is occluded, ray.tfar is set to 0.0f
    return ray.tfar == 0.0f;
}

void precomputeTransferFunctions(Mesh& mesh) {
    const int numSamples = 20000; // Increase for better accuracy
    const int numCoeffs = 9;      // 3rd-order SH

#pragma omp parallel for
    for (int v = 0; v < mesh.vertices.size(); ++v) {
        auto& vertex = mesh.vertices[v];
        std::vector<glm::vec3> shCoeffs(numCoeffs, glm::vec3(0.0f));

        // Thread-local random number generator
        std::default_random_engine engine(omp_get_thread_num() + 12345); // Seed differently for each thread
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

        for (int s = 0; s < numSamples; ++s) {
            glm::vec3 sampleDir = sampleHemisphere(vertex.Normal, engine, distribution);

            // Perform visibility test
            if (!isOccluded(vertex.Position + 0.001f * vertex.Normal, sampleDir)) {
                std::vector<float> Ylm(numCoeffs);
                computeSphericalHarmonicsBasisFunctions(sampleDir, Ylm);

                float cosTheta = glm::dot(sampleDir, vertex.Normal);
                if (cosTheta > 0.0f) {
                    glm::vec3 brdf = glm::vec3(1.0f / M_PI); // Diffuse BRDF

                    for (int i = 0; i < numCoeffs; ++i) {
                        shCoeffs[i] += brdf * Ylm[i] * cosTheta;
                    }
                }
            }
        }

        // Normalize coefficients
        for (int i = 0; i < numCoeffs; ++i) {
            shCoeffs[i] *= (M_PI / numSamples);
            vertex.shCoefficients[i] = shCoeffs[i];
        }
    }
}

void savePRTData(const Mesh& mesh, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);

    size_t numVertices = mesh.vertices.size();
    ofs.write(reinterpret_cast<const char*>(&numVertices), sizeof(size_t));

    for (const auto& vertex : mesh.vertices) {
        ofs.write(reinterpret_cast<const char*>(&vertex.Position), sizeof(glm::vec3));
        ofs.write(reinterpret_cast<const char*>(vertex.shCoefficients), sizeof(glm::vec3) * 9);
    }

    ofs.close();
}

void embreeErrorFunction(void* userPtr, enum RTCError error, const char* str) {
    std::cerr << "Embree Error: " << str << std::endl;
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create a GLFW window
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); // Request OpenGL 4.3 or newer
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL Basic Application", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable VSync to cap frame rate to monitor's refresh rate
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetScrollCallback(window, scrollCallback);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Clear any GLEW errors
    glGetError(); // Clear error flag set by GLEW

    // Enable OpenGL debugging if supported
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(MessageCallback, nullptr);

    // Optionally filter which types of messages you want to log
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

    // Define the viewport dimensions
    glViewport(0, 0, WIDTH, HEIGHT);

    glEnable(GL_DEPTH_TEST);

    glCullFace(GL_BACK); // Cull back faces (default)

    device = rtcNewDevice(nullptr);
    if (!device) {
        std::cerr << "Error initializing Embree device." << std::endl;
        return -1;
    }

    // Set error callback function (optional but recommended)
    rtcSetDeviceErrorFunction(device, embreeErrorFunction, nullptr);

    embreeScene = rtcNewScene(device);

    // Load the model
    std::vector<Mesh> meshes = loadModel(FileSystemUtils::getAssetFilePath("models/tutorial_map.obj"));

    // Build acceleration structure and precompute PRT for each mesh
    for (auto& mesh : meshes) {
        buildAccelerationStructure(mesh);
    }

    // Commit the Embree scene after attaching all geometries
    rtcCommitScene(embreeScene);

    // Precompute transfer functions
    for (auto& mesh : meshes) {
        precomputeTransferFunctions(mesh);
        mesh.setupMesh(); // Now the SH coefficients are included
        // Optionally, save the PRT data
        // savePRTData(mesh, "prt_data.bin");
    }

    // Build and compile the shader program
    // Vertex Shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    // Check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::VERTEX_SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Fragment Shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // Check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::FRAGMENT_SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Link shaders
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER_PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Original light direction from 3ds Max
    glm::vec3 lightDirMax = glm::vec3(-0.464409f, 0.513364f, -0.721652f);

    // Transform to OpenGL coordinate system
    glm::vec3 lightDirGL = glm::vec3(lightDirMax.x, lightDirMax.z, -lightDirMax.y);
    lightDirGL = glm::normalize(lightDirGL);


    // Render loop
    while (!glfwWindowShouldClose(window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        // Input handling
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        // ========== Default Forward Rendering to screen ==========
        // Bind default framebuffer (0 is the default framebuffer)
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Clear color and depth buffer
        glClearColor(0.3f, 0.3f, 0.4f, 1.0f); // Set background color
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use the regular shader program
        glUseProgram(shaderProgram);

        // Set up view and projection matrices
        glm::mat4 view = camera.getViewMatrix();
        glm::mat4 projection = camera.getProjectionMatrix((float)WIDTH / (float)HEIGHT);

        // Pass view and projection matrices to the shader
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        // Compute light SH coefficients
        glm::vec3 lightIntensity = glm::vec3(3.0f);
        std::vector<glm::vec3> lightSHCoeffs;
        computeLightSHCoefficients(lightDirGL, lightIntensity, lightSHCoeffs);

        // Pass light SH coefficients to shader
        for (int i = 0; i < 9; ++i) {
            std::string uniformName = "lightSHCoeffs[" + std::to_string(i) + "]";
            GLint location = glGetUniformLocation(shaderProgram, uniformName.c_str());
            glUniform3fv(location, 1, glm::value_ptr(lightSHCoeffs[i]));
        }

        // Render all objects
        for (const auto& mesh : meshes) {
            glm::mat4 model = glm::rotate(glm::mat4(1.0f), glm::radians(0.0f), glm::vec3(1, 0, 0));
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
            mesh.Draw(shaderProgram);
        }

        // Swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up
    glDeleteProgram(shaderProgram);

    // Clean up Embree resources
    rtcReleaseScene(embreeScene);
    rtcReleaseDevice(device);

    glfwTerminate();
    return 0;
}