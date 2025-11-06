using UnityEngine;

public class LiveEMGDisplay : MonoBehaviour
{
    public EMGInteraction emgInteraction;
    public bool displayLiveEMGValue = false;
    public bool displayLiveEMGText = false;

    private GUIStyle guiStyle = new GUIStyle();
    private Texture2D barTexture;
    private Texture2D lineTexture;


    // public float threshpos = 200;

    private void Start()
    {
        // Set font size and alignment for the label
        guiStyle.fontSize = 25;
        guiStyle.alignment = TextAnchor.MiddleCenter;

        // Create a texture for the bar
        barTexture = new Texture2D(1, 1);
        barTexture.SetPixel(0, 0, Color.green);
        barTexture.Apply();

        // Create a texture for the threshold line
        lineTexture = new Texture2D(1, 1);
        lineTexture.SetPixel(0, 0, Color.red);
        lineTexture.Apply();
    }

    private void OnGUI()
    {
        if (displayLiveEMGValue)
        {
            if (displayLiveEMGText)
            {
                GUILayout.Label("Live EMG Value: " + emgInteraction.LiveEMGValue(0), guiStyle);
            }
            // Calculate the percentage of the current EMG value relative to the maximum value of 1023
            float emgPercent = Mathf.Clamp(emgInteraction.LiveEMGValue(0) / 1023f, 0f, 1f);

           

            // Draw the bar with the calculated percentage
            Rect barRect = new Rect(100, 50, 400 * emgPercent, 25);
            GUI.DrawTexture(barRect, barTexture);

            // Calculate the percentage of the EMG threshold value relative to the maximum value of 1023
            float threshPercent = Mathf.Clamp(emgInteraction.EMGThreshold / 1023f, 0f, 1f);

            Rect threshRect = new Rect(100 + 400 * threshPercent, 50,  5, 25);
            GUI.DrawTexture(threshRect, lineTexture);

        }

    }

    public void EMGboolswap()
    {
        if (displayLiveEMGValue == true)
            displayLiveEMGValue = false;

        else if (displayLiveEMGValue == false)
            displayLiveEMGValue = true;
    }

}
