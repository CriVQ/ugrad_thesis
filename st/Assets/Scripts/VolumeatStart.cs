using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;


public class VolumeatStart : MonoBehaviour
{

    public VolumeProfile postProcessingProfile;
    private Vignette vignette;
    public float _fadedur = 1.0f;
    // Start is called before the first frame update
    void Start()
    {
        if (postProcessingProfile.TryGet(out vignette))
        {
            // You can also modify other vignette properties here, like color, smoothness, etc.
            vignette.intensity.value = 0.0f; // Start with the current intensity value
            StartCoroutine(FadeIn());
        }
        else
        {
            Debug.LogError("Vignette not found in the Post-Processing Profile.");
        }
    }


    private IEnumerator FadeIn()
    {
        float elapsedTime = 0f;

        while (elapsedTime < _fadedur)
        {
            elapsedTime += Time.deltaTime;
            float t = Mathf.Clamp01(elapsedTime / _fadedur);
            vignette.intensity.value = Mathf.Lerp(1.0f, 0.0f, t) ;
            yield return null;
        }

        vignette.intensity.value = 0;
    }

    public void OnSceneChange()
    {
        vignette.intensity.value = 0.0f;
        StartCoroutine(FadeOut());
    }

    private IEnumerator FadeOut()
    {
        float elapsedTime = 0f;

        while (elapsedTime < _fadedur)
        {
            elapsedTime += Time.deltaTime;
            float t = Mathf.Clamp01(elapsedTime / _fadedur);
            vignette.intensity.value = Mathf.Lerp(0.0f, 1.0f, t);
            yield return null;
        }

        vignette.intensity.value = 1;
    }


}
