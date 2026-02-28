using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using BepInEx;
using UnityEngine;

namespace HKTelemetryMod
{
    [BepInPlugin("com.dayya.hollowknight.telemetry", "HK Telemetry Mod", "1.0.0")]
    public class HKTelemetryPlugin : BaseUnityPlugin
    {
        private UdpClient udpClient;
        private IPEndPoint endPoint;
        private HealthManager currentBoss;

        private void Start()
        {
            Logger.LogInfo("HKTelemetry Mod Initializing...");

            udpClient = new UdpClient();
            endPoint = new IPEndPoint(IPAddress.Parse("127.0.0.1"), 5005);
        }

        private void Update()
        {
            try
            {
                int playerHp = PlayerData.instance != null ? PlayerData.instance.health : 0;

                if (currentBoss == null || !currentBoss.gameObject.activeInHierarchy)
                {
                    HealthManager[] hms = UnityEngine.Object.FindObjectsOfType<HealthManager>();
                    foreach (HealthManager hm in hms)
                    {
                        if (hm.hp >= 100)
                        {
                            currentBoss = hm;
                            break;
                        }
                    }
                }

                int bossHp = currentBoss != null ? currentBoss.hp : 0;

                string json = $"{{\"player_hp\": {playerHp}, \"boss_hp\": {bossHp}}}";
                byte[] bytes = Encoding.UTF8.GetBytes(json);
                udpClient.Send(bytes, bytes.Length, endPoint);
            }
            catch (Exception)
            {}
        }
    }
}
