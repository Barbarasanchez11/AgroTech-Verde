import os
from dotenv import load_dotenv
from supabase import create_client, Client


load_dotenv('.streamlit/secrets.toml')

def test_supabase_connection():
    try:
     
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_ANON_KEY')
        
        print(f"URL: {url}")
        print(f"Key: {key[:20]}..." if key else "Key: None")
        
        if not url or not key:
            print("❌ Faltan credenciales en .streamlit/secrets.toml")
            return False
        
        # Crear cliente
        supabase: Client = create_client(url, key)
        print("✅ Cliente Supabase creado")
        
        # Probar conexión - obtener datos de la tabla crops
        result = supabase.table('crops').select('*').limit(1).execute()
        print(f"✅ Conexión exitosa - Tabla crops accesible")
        print(f"   Datos obtenidos: {len(result.data)} registros")
        
        return True
        
    except Exception as e:
        print(f"❌ Error conectando a Supabase: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Probando conexión a Supabase...")
    success = test_supabase_connection()
    
    if success:
        print("\n🎉 ¡Conexión exitosa! Supabase está funcionando correctamente.")
    else:
        print("\n💥 Error en la conexión. Revisa las credenciales.") 